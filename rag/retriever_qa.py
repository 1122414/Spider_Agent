import os
import sys
import torch
import httpx
import traceback
from typing import List, Tuple
from dotenv import load_dotenv

# LangChain 相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Transformers 相关 (Qwen Reranker)
from transformers import AutoTokenizer, AutoModelForCausalLM

# 自定义模块 (假设这些在你的项目中存在)
from config import * 
from rag.query_analyzer import query_analyzer
from agent.prompt_template import RAG_PROMPT

load_dotenv()

# ==============================================================================
# 1. 配置区域
# ==============================================================================
# 显存优化配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERANK_MODEL_PATH = r"G:\models\Qwen\Qwen3-Reranker-4B"
RERANK_MAX_LENGTH = 8192

# ==============================================================================
# 2. 核心类：QwenReranker (封装官方逻辑)
# ==============================================================================

class QwenReranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"🚀 [System] Loading Qwen3-Reranker-4B on {DEVICE}...")
            cls._instance = super(QwenReranker, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                RERANK_MODEL_PATH, 
                padding_side='left', 
                trust_remote_code=True
            )
            
            # 自动选择精度，显存充足建议 float16，且使用 flash_attention_2
            model_kwargs = {
                "device_map": DEVICE,
                "trust_remote_code": True
            }
            if DEVICE == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                # 如果安装了 flash-attn 库，取消下面注释以获得加速
                # model_kwargs["attn_implementation"] = "flash_attention_2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                RERANK_MODEL_PATH, 
                **model_kwargs
            ).eval()

            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            # Prompt 模板构建块
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        except Exception as e:
            print(f"❌ [Error] Failed to load Qwen Reranker: {e}")
            raise e

    def _format_instruction(self, query: str, doc_content: str, instruction: str = None) -> str:
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc_content
        )

    def _process_inputs(self, pairs: List[str]):
        # 官方逻辑：手动拼接 Token 并处理 Padding
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=RERANK_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # 拼接 prefix 和 suffix
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        # Pad 到同一长度
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=RERANK_MAX_LENGTH)
        
        # 移动到 GPU
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
            
        return inputs

    @torch.no_grad()
    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """
        核心重排序方法
        """
        if not docs:
            return []

        # 1. 准备输入对
        doc_contents = [doc.page_content for doc in docs]
        pairs = [self._format_instruction(query, content) for content in doc_contents]

        # 2. Tokenize & Process
        inputs = self._process_inputs(pairs)

        # 3. 推理 (Compute Logits)
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        # Stack & Log Softmax
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        # 获取 "yes" 的概率作为分数
        scores = batch_scores[:, 1].exp().tolist()

        # 4. 排序并组合结果
        doc_score_pairs = list(zip(docs, scores))
        # 按分数降序排列
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"📊 [Rerank] Top score: {doc_score_pairs[0][1]:.4f} | Low score: {doc_score_pairs[-1][1]:.4f}")
        
        # 返回 Top K 文档
        return [doc for doc, _ in doc_score_pairs[:top_k]]

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

def get_embedding_model():
    """自动选择 OpenAI 或 Ollama Embeddings"""
    http_client = httpx.Client(trust_env=False, timeout=60.0)
    if EMBEDDING_TYPE == 'local_ollama':
        print(f"🔌 使用 OllamaEmbeddings (Model: {OPENAI_OLLAMA_EMBEDDING_MODEL})...")
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/api/generate", "").replace("/v1", "").rstrip("/")
        return OllamaEmbeddings(base_url=base_url, model=OPENAI_OLLAMA_EMBEDDING_MODEL)
    elif EMBEDDING_TYPE == 'local_vllm':
        return OpenAIEmbeddings(
            model=VLLM_OPENAI_EMBEDDING_MODEL,
            openai_api_key=VLLM_OPENAI_EMBEDDING_API_KEY,
            openai_api_base=VLLM_OPENAI_EMBEDDING_BASE_URL,
            http_client=http_client,
            check_embedding_ctx_length=False
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_OLLAMA_BASE_URL
        )

def format_docs(docs):
    return "\n\n".join(f"[片段 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

def get_retrieval_k(question: str) -> int:
    global_keywords = ["全部", "所有", "列表", "清单", "总结", "分析", "all", "summary"]
    if any(kw in question.lower() for kw in global_keywords):
        return 20 # 总结类问题需要更多上下文
    return 10     # 事实类问题

# ==============================================================================
# 4. 主业务逻辑 (Hybrid: Vector Search + Qwen Rerank)
# ==============================================================================

def qa_interaction(question: str) -> str:
    print(f"🤔 RAG Searching for: {question}")
    
    embeddings = get_embedding_model()
    generated_expr = query_analyzer.generate_expr(question)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )

    try:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        # --- Step 1: 连接 Milvus ---
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            consistency_level="Bounded",
            auto_id=True,
        )
        
        # --- Step 2: 向量粗排 (Recall) ---
        target_k = get_retrieval_k(question)
        # 扩大召回池，给 Reranker 足够的选择空间 (建议 5-10 倍 target_k)
        recall_k = target_k * 5 
        
        print(f"🔍 [Retrieve] Fetching Top-{recall_k} candidates from Milvus...")
        
        # 使用 MMR 增加多样性，防止召回过于相似的内容
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": recall_k,
                "expr": generated_expr,
                "fetch_k": recall_k * 2,
                "lambda_mult": 0.6
            }
        )
        
        # 显式执行检索 (LCEL Chain 很难插入重排序，所以这里断开 Chain 手动执行)
        initial_docs = retriever.invoke(question)
        if not initial_docs:
            return "没有在知识库中找到相关信息。"

        # --- Step 3: 精排 (Rerank with Qwen) ---
        print(f"⚖️ [Rerank] Re-ranking {len(initial_docs)} docs using Qwen3-Reranker-4B...")
        reranker = QwenReranker() # 获取单例
        final_docs = reranker.rerank(question, initial_docs, top_k=target_k)

        # --- Step 4: 生成 (Generate) ---
        if RAG_PROMPT:
            if isinstance(RAG_PROMPT, str):
                custom_rag_prompt = PromptTemplate.from_template(RAG_PROMPT)
            else:
                custom_rag_prompt = RAG_PROMPT
        else:
            template = """基于以下上下文回答问题。如果你不知道答案，请直接说不知道。\n\n上下文：\n{context}\n\n问题：{question}"""
            custom_rag_prompt = PromptTemplate.from_template(template)

        # 构建最终上下文
        formatted_context = format_docs(final_docs)
        
        # 手动执行 Chain 的最后一步
        chain = (
            custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({"context": formatted_context, "question": question})
        return response

    except Exception as e:
        traceback.print_exc()
        return f"RAG 系统出错: {str(e)}"

if __name__ == "__main__":
    # 简单的命令行测试
    q = sys.argv[1] if len(sys.argv) > 1 else "测试：介绍一下 Qwen Reranker 的优势？"
    print(qa_interaction(q))
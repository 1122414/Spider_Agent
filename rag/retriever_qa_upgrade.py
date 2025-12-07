import os
import sys
import torch
import httpx
import traceback
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# --- LangChain Core ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Embedding Models ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# --- Milvus Native Client ---
from pymilvus import (
    connections, 
    Collection, 
    AnnSearchRequest, 
    RRFRanker, 
    utility
)

# --- Transformers (Rerank) ---
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 项目配置 ---
from config import *
try:
    from rag.query_analyzer import query_analyzer
except ImportError:
    query_analyzer = None
try:
    from agent.prompt_template import RAG_PROMPT
except ImportError:
    RAG_PROMPT = None

load_dotenv()

# ==============================================================================
# 0. 全局配置
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not os.environ.get("RERANK_MODEL_PATH"):
    # 默认回退
    RERANK_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct" 
else:
    RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH")
    
RERANK_MAX_LENGTH = 2048

# ==============================================================================
# 1. 辅助函数：Embedding 模型工厂
# ==============================================================================
def get_embedding_model():
    """获取 LangChain Embedding 实例"""
    http_client = httpx.Client(trust_env=False, timeout=60.0)
    
    if EMBEDDING_TYPE == 'local_ollama':
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

# TODO: 这里必须与 ingest_tool.py 使用相同的 Sparse 模型/逻辑
def _get_query_sparse_vector(text: str) -> Dict[int, float]:
    """生成查询的稀疏向量 (Mock)"""
    sparse_vec = {}
    for char in text[:50]: 
        token_id = abs(hash(char)) % 10000 
        sparse_vec[token_id] = sparse_vec.get(token_id, 0.0) + 0.1
    return sparse_vec

# ==============================================================================
# 2. 核心类：QwenReranker
# ==============================================================================
class QwenReranker:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            print(f"🚀 [System] Loading Qwen Reranker on {DEVICE}...")
            cls._instance = super(QwenReranker, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH, padding_side='left', trust_remote_code=True)
            model_kwargs = {"device_map": DEVICE, "trust_remote_code": True}
            if DEVICE == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL_PATH, **model_kwargs).eval()
            
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Answer 'yes' or 'no'.<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n"
            
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        except Exception as e:
            print(f"❌ [Error] Failed to load Reranker: {e}")
            self.model = None

    def _format_input(self, query: str, doc_content: str) -> str:
        return f"Query: {query}\nDocument: {doc_content[:1000]}"

    @torch.no_grad()
    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs or not self.model: return docs[:top_k]
        
        pairs = [self._format_input(query, doc.page_content) for doc in docs]
        
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=RERANK_MAX_LENGTH, 
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        scores = logits[:, self.token_true_id].float().cpu().numpy()
        
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 [Rerank] Top score: {doc_score_pairs[0][1]:.4f}")
        return [doc for doc, _ in doc_score_pairs[:top_k]]

# ==============================================================================
# 3. 核心功能：Milvus 原生混合检索 (Hybrid Search)
# ==============================================================================

def execute_hybrid_search(collection_name: str, query: str, expr: str = "", top_k: int = 10) -> List[Document]:
    """
    执行 Milvus 混合检索 (Dense + Sparse) + RRF 融合
    """
    # 1. Connect
    try:
        connections.connect(alias="default", uri=MILVUS_URI)
        if not utility.has_collection(collection_name):
            print(f"⚠️ Collection {collection_name} does not exist.")
            return []
        
        col = Collection(collection_name)
        col.load()
    except Exception as e:
        print(f"❌ Milvus Connect Error: {e}")
        return []

    # 2. Prepare Vectors
    # A. Dense Query Vector
    embeddings = get_embedding_model()
    dense_vector = embeddings.embed_query(query)
    
    # B. Sparse Query Vector (Mock or Real)
    sparse_vector = _get_query_sparse_vector(query)

    # 3. Build Requests
    search_requests = []
    
    # --- 路 A: Dense Request ---
    req_dense = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector", 
        param={"metric_type": "IP", "params": {"ef": 128}}, 
        limit=top_k * 3, 
        expr=expr
    )
    search_requests.append(req_dense)

    # --- 路 B: Sparse Request ---
    req_sparse = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, # Sparse Search Params
        limit=top_k * 3,
        expr=expr
    )
    search_requests.append(req_sparse)

    # 4. Execute Hybrid Search
    print(f"🔍 [Hybrid] Executing multi-path search (Dense + Sparse)...")
    try:
        # 使用 RRF (Reciprocal Rank Fusion) 进行融合
        # k=60 是经典参数，平滑了不同路分数的差异
        results = col.hybrid_search(
            reqs=search_requests,
            rerank=RRFRanker(k=60), 
            limit=top_k * 2, 
            output_fields=["text", "title", "source", "type", "category"]
        )
    except Exception as e:
        print(f"❌ Search Execution Failed: {e}")
        return []

    # 5. Parse Results
    documents = []
    if not results: return []
        
    for hits in results:
        for hit in hits:
            entity = hit.entity
            text = entity.get("text") or ""
            meta = {
                "id": hit.id,
                "score": hit.score,
                "title": entity.get("title"),
                "source": entity.get("source"),
                "category": entity.get("category"),
                "type": entity.get("type")
            }
            documents.append(Document(page_content=text, metadata=meta))
            
    print(f"   -> Found {len(documents)} candidates.")
    return documents

# ==============================================================================
# 4. 主入口
# ==============================================================================

def format_docs(docs):
    return "\n\n".join(f"[Result {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

def qa_interaction(question: str) -> str:
    print(f"\n🤔 [RAG] Question: {question}")
    
    # 1. Analyze Intent
    expr = ""
    if query_analyzer:
        expr = query_analyzer.generate_expr(question)
        if expr: print(f"   🎯 Metadata Filter: {expr}")

    # 2. Retrieve
    target_k = 10
    if "全部" in question or "list" in question.lower(): target_k = 20
        
    initial_docs = execute_hybrid_search(COLLECTION_NAME, question, expr, target_k)

    if not initial_docs:
        return "数据库中没有找到相关信息。"

    # 3. Rerank
    print(f"⚖️ [Rerank] Refining results...")
    reranker = QwenReranker()
    if reranker.model:
        final_docs = reranker.rerank(question, initial_docs, top_k=target_k)
    else:
        final_docs = initial_docs[:target_k]

    # 4. Generate
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    
    if RAG_PROMPT:
        prompt_tmpl = PromptTemplate.from_template(RAG_PROMPT)
    else:
        prompt_tmpl = PromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\nAnswer:")

    chain = prompt_tmpl | llm | StrOutputParser()
    
    print("📝 [Generate] Synthesizing answer...")
    response = chain.invoke({
        "context": format_docs(final_docs), 
        "question": question
    })
    
    return response

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "测试：介绍一下系统内容"
    print(qa_interaction(q))
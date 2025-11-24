import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import *

# 尝试导入自定义 prompt_template
try:
    from agent.prompt_template import RAG_PROMPT
except ImportError:
    RAG_PROMPT = None

load_dotenv()

# ==============================================================================
# 1. 配置区域
# ==============================================================================
# MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
# COLLECTION_NAME = "spider_knowledge_base"

# # Embedding 配置
# EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
# MODEL_NAME = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")
# OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
# OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")

# # 本地 Ollama
# OPENAI_OLLAMA_EMBEDDING_MODEL = os.environ.get("OPENAI_OLLAMA_EMBEDDING_MODEL", "text-embedding-3-small")
# OPENAI_OLLAMA_BASE_URL = os.environ.get("OPENAI_OLLAMA_BASE_URL", OPENAI_BASE_URL)

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def get_embedding_model():
    """
    自动选择 OpenAI 或 Ollama Embeddings
    增加了对 Ollama URL 的容错处理
    """
    # 优先检查是否存在 Ollama 的特征端口 11434
    
    # 本地Ollama框架
    if EMBEDDING_TYPE == 'local_ollama':
        print(f"🔌 使用 OllamaEmbeddings (Model: {OPENAI_OLLAMA_EMBEDDING_MODEL})...")
        # OllamaEmbeddings 不需要 /v1 后缀
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/api/generate", "").replace("/v1", "").rstrip("/")
        return OllamaEmbeddings(
            base_url=base_url,
            model=OPENAI_OLLAMA_EMBEDDING_MODEL
        )
    elif EMBEDDING_TYPE == 'local_vllm':
        print(f"🔌 使用 Vllm OpenAIEmbeddings (Model: {VLLM_OPENAI_EMBEDDING_MODEL})...")
        return OpenAIEmbeddings(
            model=VLLM_OPENAI_EMBEDDING_MODEL,
            openai_api_key=VLLM_OPENAI_EMBEDDING_API_KEY,
            openai_api_base=VLLM_OPENAI_EMBEDDING_BASE_URL,
            # 关闭本地 Token 检查，强制发送纯文本
            check_embedding_ctx_length=False
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_OLLAMA_BASE_URL
        )

def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(f"[片段 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

def get_retrieval_k(question: str) -> int:
    """
    根据问题类型动态调整检索数量 (Top-K)
    """
    global_keywords = ["全部", "所有", "列表", "清单", "总结", "分析", "概括", "all", "summary", "list", "有哪些", "统计"]
    
    if any(kw in question.lower() for kw in global_keywords):
        print("🚀 [Config] 全局/总结性提问 -> 检索 Top-20")
        return 20
    else:
        print("🔍 [Config] 事实性提问 -> 检索 Top-10")
        return 10

# ==============================================================================
# 3. 主业务逻辑 (纯向量检索版)
# ==============================================================================

def qa_interaction(question: str) -> str:
    print(f"🤔 RAG Searching for: {question}")
    
    embeddings = get_embedding_model()
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )

    try:
        # 1. 连接 Milvus 向量库
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )
        
        # 2. 确定检索数量
        top_k = get_retrieval_k(question)
        
        # 3. 获取检索器
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        
        # 4. 准备 Prompt
        if RAG_PROMPT:
            if isinstance(RAG_PROMPT, str):
                custom_rag_prompt = PromptTemplate.from_template(RAG_PROMPT)
            else:
                custom_rag_prompt = RAG_PROMPT
        else:
            template = """基于以下上下文回答问题。如果你不知道答案，请直接说不知道。\n\n上下文：\n{context}\n\n问题：{question}"""
            custom_rag_prompt = PromptTemplate.from_template(template)

        # 5. 构建并执行 Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"RAG 系统出错: {str(e)}"

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "测试：告诉我当前知识库里有什么内容？"
    print(qa_interaction(q))
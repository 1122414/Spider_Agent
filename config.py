import os
from dotenv import load_dotenv

# 1. 在这里统一加载 .env，其他文件就不需要再写 load_dotenv() 了
load_dotenv()

# ==========================
# 基础配置
# ==========================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "spider_knowledge_base"

# ==========================
# 魔搭模型参数配置（不加Ollama的全是魔搭、线上API）
# ==========================
EMBEDDING_MODEL = os.getenv("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
MODEL_NAME = os.getenv("MODA_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("MODA_OPENAI_BASE_URL")

# ==========================
# 本地Ollama
# ==========================
OPENAI_OLLAMA_EMBEDDING_MODEL = os.getenv("OPENAI_OLLAMA_EMBEDDING_MODEL", OPENAI_BASE_URL)
OPENAI_OLLAMA_BASE_URL = os.getenv("OPENAI_OLLAMA_BASE_URL")

EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "api").lower()

if OPENAI_BASE_URL:
    # 统一清洗逻辑：去除 /api/generate, /v1, 尾部斜杠
    base_url = OPENAI_OLLAMA_BASE_URL.replace("/v1", "").strip("/")

# ==========================
# 服务器Vllm
# ==========================
VLLM_OPENAI_EMBEDDING_MODEL = os.getenv("VLLM_OPENAI_EMBEDDING_MODEL")
VLLM_OPENAI_EMBEDDING_API_KEY = os.getenv("VLLM_OPENAI_EMBEDDING_API_KEY")
VLLM_OPENAI_EMBEDDING_BASE_URL = os.getenv("VLLM_OPENAI_EMBEDDING_BASE_URL")

# Rerank 配置
RERANK_TYPE = os.getenv("RERANK_TYPE", "api").lower()
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH")

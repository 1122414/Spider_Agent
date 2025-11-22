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
# 模型参数配置
# ==========================
EMBEDDING_MODEL = os.getenv("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
MODEL_NAME = os.getenv("MODA_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("MODA_OPENAI_BASE_URL")

# ==========================
# 特殊逻辑处理 (比如 URL 清洗)
# 这里写一次，所有文件都受益
# ==========================
raw_ollama_url = os.getenv("MODA_OLLAMA_BASE_URL", OPENAI_BASE_URL)
OPENAI_OLLAMA_BASE_URL = None

if raw_ollama_url:
    # 统一清洗逻辑：去除 /api/generate, /v1, 尾部斜杠
    OPENAI_OLLAMA_BASE_URL = raw_ollama_url.replace("/api/generate", "").replace("/v1", "").rstrip("/")

# Rerank 配置
RERANK_TYPE = os.getenv("RERANK_TYPE", "api").lower()

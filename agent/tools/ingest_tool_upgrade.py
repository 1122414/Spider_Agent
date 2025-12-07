import os
import time
import json
import uuid
import traceback
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# Milvus Native SDK
from pymilvus import (
    connections,
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    Collection,
    MilvusException
)

# LangChain Embeddings (仅用于生成稠密向量)
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# 引入 registry
from agent.tools.registry import tool_registry
from config import *

load_dotenv()

# ========================== 配置区域 ==========================
# 向量维度需与 Embedding 模型一致 (OpenAI text-embedding-3-small 默认为 1536)
VECTOR_DIM = 4096
DENSE_FIELD_NAME = "dense_vector"
SPARSE_FIELD_NAME = "sparse_vector"

def get_embedding_model():
    """工厂函数：自动选择 OpenAI 或 Ollama 嵌入模型"""
    if EMBEDDING_TYPE == 'local_ollama':
        print(f"🔌 使用 OllamaEmbeddings (Model: {OPENAI_OLLAMA_EMBEDDING_MODEL})...")
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/api/generate", "").replace("/v1", "").rstrip("/")
        return OllamaEmbeddings(base_url=base_url, model=OPENAI_OLLAMA_EMBEDDING_MODEL)
    elif EMBEDDING_TYPE == 'local_vllm':
        print(f"🔌 使用 Vllm OpenAIEmbeddings (Model: {VLLM_OPENAI_EMBEDDING_MODEL})...")
        return OpenAIEmbeddings(
            model=VLLM_OPENAI_EMBEDDING_MODEL,
            openai_api_key=VLLM_OPENAI_EMBEDDING_API_KEY,
            openai_api_base=VLLM_OPENAI_EMBEDDING_BASE_URL,
            check_embedding_ctx_length=False
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_OLLAMA_BASE_URL
        )

# TODO: 这里需要接入真正的稀疏向量模型 (如 BGE-M3, SPLADE)
# 目前仅返回伪造数据或空字典以跑通流程
def _get_sparse_vector(text: str) -> Dict[int, float]:
    """
    生成稀疏向量 (Sparse Vector)。
    格式: {word_id: weight, ...} 或 Milvus 接受的稀疏格式
    """
    # 模拟：简单的 Hash 词频 (仅用于测试 Pipeline，无实际检索意义)
    # 生产环境请替换为: splade_model.encode(text)
    sparse_vec = {}
    # 简单的伪造逻辑：取部分字符的 hash 作为 ID
    for char in text[:50]: 
        token_id = abs(hash(char)) % 10000 
        sparse_vec[token_id] = sparse_vec.get(token_id, 0.0) + 0.1
    return sparse_vec

def _init_collection(collection_name: str, drop_old: bool = False) -> Collection:
    """初始化 Milvus Collection (Define Schema & Index)"""
    try:
        connections.connect(alias="default", uri=MILVUS_URI)
        print(f"🔌 Connected to Milvus at {MILVUS_URI}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        raise e

    if utility.has_collection(collection_name) and drop_old:
        print(f"🗑️ Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    if utility.has_collection(collection_name):
        print(f"✅ Collection {collection_name} exists. Loading...")
        return Collection(collection_name)

    print(f"🆕 Creating new collection: {collection_name}...")
    
    # --- 1. Define Schema ---
    fields = [
        # 主键 (Auto ID)
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 稠密向量 (Dense Vector) - 用于语义检索
        FieldSchema(name=DENSE_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        # 稀疏向量 (Sparse Vector) - 用于关键词/混合检索 (Milvus 2.4+)
        FieldSchema(name=SPARSE_FIELD_NAME, dtype=DataType.SPARSE_FLOAT_VECTOR), 
        # 元数据字段
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # 原始内容
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=64), # parent_info / child_detail
        FieldSchema(name="crawled_at", dtype=DataType.INT64) # Timestamp
    ]
    
    schema = CollectionSchema(fields, description="Spider Agent Knowledge Base (Hybrid Ready)")
    col = Collection(name=collection_name, schema=schema)

    # --- 2. Create Indexes (关键步骤) ---
    print("🔨 Building Indexes...")
    
    # A. 稠密向量索引 (HNSW)
    dense_index_params = {
        "metric_type": "IP", # Inner Product
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index(field_name=DENSE_FIELD_NAME, index_params=dense_index_params)
    
    # B. 稀疏向量索引 (SPARSE_INVERTED_INDEX)
    sparse_index_params = {
        "metric_type": "IP",
        "index_type": "SPARSE_INVERTED_INDEX",
        "params": {"drop_ratio_build": 0.2} # 过滤掉权重过小的项，减小索引体积
    }
    col.create_index(field_name=SPARSE_FIELD_NAME, index_params=sparse_index_params)

    # C. 标量字段索引 (加速 Metadata Filter)
    try:
        col.create_index(field_name="category", index_name="idx_category")
        col.create_index(field_name="type", index_name="idx_type")
    except Exception as e:
        print(f"   (Scalar index warning: {e})")

    print("✅ Collection initialized successfully.")
    return col

def _resolve_data(data: Union[Dict, List, None]) -> Union[Dict, List, None]:
    cached_data = tool_registry.last_execution_result
    if cached_data:
        if isinstance(cached_data, str):
            print(f"⚠️ [SaveToKB] 缓存数据是字符串，跳过。")
        elif isinstance(cached_data, list) and len(cached_data) == 0:
            print("⚠️ [SaveToKB] 缓存数据为空。")
            return cached_data
        else:
            print(f"🔎 [SaveToKB] 使用内存缓存数据。")
            return cached_data
    return data if data else None

def _flatten_data_to_payloads(data: Union[List, Dict], category: str) -> List[Dict]:
    """
    将嵌套数据扁平化为适合插入 Milvus 的字典列表
    """
    items = []
    if isinstance(data, list): items = data
    elif isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list): items = data["items"]
        else: items = [data]
    
    if not items: return []

    payloads = []
    timestamp = int(time.time())

    for item in items:
        if not isinstance(item, dict): continue
        
        # 提取基础信息
        url = item.get("url") or item.get("link") or ""
        title = item.get("title") or "未命名条目"
        
        # 1. 构建 Parent Text
        content_parts = []
        for k, v in item.items():
            if k in ["children", "target_content", "items"] or v is None: continue
            val_str = str(v).strip()
            if val_str: content_parts.append(f"{k}: {val_str}")
        
        parent_text = "\n".join(content_parts)
        if len(parent_text) > 5:
            payloads.append({
                "text": parent_text,
                "source": url,
                "title": title,
                "category": category,
                "type": "parent_info",
                "crawled_at": timestamp
            })

        # 2. 构建 Children Text
        children = item.get("children", [])
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict): continue
                child_parts = []
                for k, v in child.items():
                    val_str = str(v).strip()
                    if val_str: child_parts.append(f"{k}: {val_str}")
                
                if child_parts:
                    child_text = f"《{title}》详情:\n" + "\n".join(child_parts)
                    payloads.append({
                        "text": child_text,
                        "source": url,
                        "title": title,
                        "category": category,
                        "type": "child_detail",
                        "crawled_at": timestamp
                    })
    return payloads

def save_to_milvus(data: Union[Dict, List] = None, category: str = "general") -> str:
    """
    将数据存入 Milvus (Hybrid Ready)
    """
    actual_data = _resolve_data(data)
    if not actual_data: return "保存失败: 没有有效数据"

    # 1. 数据清洗与扁平化
    payloads = _flatten_data_to_payloads(actual_data, category)
    if not payloads: return "保存失败: 数据解析后为空"
    
    print(f"🔄 准备入库 {len(payloads)} 条数据 (Category: {category})...")

    try:
        # 2. 初始化 Collection (Drop Old=False 默认保留数据)
        col = _init_collection(COLLECTION_NAME, drop_old=False) 

        # 3. 计算向量 (Dense Embedding)
        embeddings_model = get_embedding_model()
        texts = [p["text"] for p in payloads]
        
        print(f"⚡ 计算 Dense Vectors (Batch: {len(texts)})...")
        dense_vectors = embeddings_model.embed_documents(texts)
        
        # 4. 计算稀疏向量 (Sparse Embedding)
        print(f"⚡ 计算 Sparse Vectors (Mock)...")
        sparse_vectors = [_get_sparse_vector(t) for t in texts]
        
        if len(dense_vectors) != len(payloads):
            return "保存失败: 向量数量与文本数量不匹配"

        # 5. 组装 Insert Data (Column-based for PyMilvus)
        # 顺序必须严格对应 Schema 定义: 
        # [dense_vector, sparse_vector, text, source, title, category, type, crawled_at]
        entities = [
            dense_vectors,                              # dense_vector
            sparse_vectors,                             # sparse_vector
            [p["text"][:60000] for p in payloads],      # text
            [str(p["source"])[:1000] for p in payloads],# source
            [str(p["title"])[:1000] for p in payloads], # title
            [str(p["category"]) for p in payloads],     # category
            [p["type"] for p in payloads],              # type
            [p["crawled_at"] for p in payloads]         # crawled_at
        ]

        # 6. 执行插入
        print("💾 正在写入 Milvus...")
        insert_res = col.insert(entities)
        
        # 7. Flush (确保数据可见)
        col.flush() 
        
        cnt = insert_res.insert_count
        return f"✅ 成功入库 {cnt} 条数据 (Collection: {COLLECTION_NAME})"

    except Exception as e:
        traceback.print_exc()
        return f"❌ 向量数据库操作失败: {str(e)}"
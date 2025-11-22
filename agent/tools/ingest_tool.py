import os
import time
import json
import random
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# LangChain Components
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings # 新增 Ollama 支持
from langchain_milvus import Milvus

# 引入 registry 以获取缓存数据
from agent.tools.registry import tool_registry

load_dotenv()

# ========================== 配置区域 ==========================
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "spider_knowledge_base"

# Embedding 配置
EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")

# 本地 Ollama
OPENAI_OLLAMA_EMBEDDING_MODEL = os.environ.get("OPENAI_OLLAMA_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_OLLAMA_BASE_URL = os.environ.get("OPENAI_OLLAMA_BASE_URL", OPENAI_BASE_URL)

def get_embedding_model():
    """
    工厂函数：自动选择 OpenAI 或 Ollama 嵌入模型
    """
    # 简单的自动判定逻辑：如果 Base URL 包含 11434 (Ollama 默认端口)，则使用 OllamaEmbeddings
    if OPENAI_OLLAMA_BASE_URL and "11434" in OPENAI_OLLAMA_BASE_URL:
        print(f"🔌 检测到本地 Ollama 环境，切换至 OllamaEmbeddings (Model: {EMBEDDING_MODEL})...")
        # OllamaEmbeddings 不需要 /v1 后缀
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/v1", "").strip("/")
        return OllamaEmbeddings(
            base_url=base_url,
            model=EMBEDDING_MODEL
        )
    else:
        # 默认使用 OpenAI 兼容模式
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL
        )

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

def _flatten_data_to_documents(data: Union[List, Dict]) -> List[Document]:
    items = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("items") or data.get("target_content") or []
        if not items and "url" in data: items = [data]
    elif isinstance(data, list):
        items = data
        
    if not items: return []

    documents = []
    for item in items:
        if not isinstance(item, dict): continue
        
        title = item.get("电影名称") or item.get("title") or item.get("name") or "未知标题"
        url = item.get("链接") or item.get("url") or item.get("link") or ""
        
        content_parts = []
        for k, v in item.items():
            if k not in ["children", "url", "link", "href", "跳转链接"] and v and isinstance(v, str) and v.strip():
                content_parts.append(f"{k}: {v.strip()}")
        
        parent_text = "\n".join(content_parts)
        
        if parent_text and len(parent_text.strip()) > 5:
            doc = Document(
                page_content=parent_text,
                metadata={"source": url, "title": title, "type": "parent_info"}
            )
            documents.append(doc)
            
        children = item.get("children", [])
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict): continue
                child_parts = []
                for k, v in child.items():
                    if v and isinstance(v, str) and v.strip():
                        child_parts.append(f"{k}: {v.strip()}")
                
                if child_parts:
                    child_text = f"《{title}》的详细信息:\n" + "\n".join(child_parts)
                    if child_text and len(child_text.strip()) > 5:
                        child_doc = Document(
                            page_content=child_text,
                            metadata={"source": url, "title": title, "type": "child_detail"}
                        )
                        documents.append(child_doc)
    return documents

def save_to_milvus(data: Union[Dict, List] = None) -> str:
    """
    将数据存入 Milvus 向量知识库 (支持 OpenAI/Ollama)
    """
    actual_data = _resolve_data(data)
    if not actual_data:
        return "保存失败: 没有有效数据"

    docs = _flatten_data_to_documents(actual_data)
    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    
    if not valid_docs:
        return "保存失败: 数据转换后为空"
    
    print(f"🔄 准备处理 {len(valid_docs)} 条数据片段...")

    try:
        # 1. 初始化 Embedding 模型 (自动判断类型)
        embeddings = get_embedding_model()

        # 2. 手动计算向量 (带重试机制)
        text_embeddings = []
        metadatas = []
        texts = []
        
        print(f"⚡ 开始计算向量 (Manual Embedding Mode)...")
        for i, doc in enumerate(valid_docs):
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    clean_text = doc.page_content.replace("\n", " ")
                    
                    # embed_query 是所有 LangChain Embedding 类都支持的标准接口
                    vector = embeddings.embed_query(clean_text)
                    
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)
                    text_embeddings.append(vector)
                    
                    if (i + 1) % 5 == 0:
                        print(f"   -> 已向量化 {i + 1}/{len(valid_docs)} 条")
                    
                    # 本地模型不需要 sleep，但为了保险起见保留微小延迟
                    time.sleep(1)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    # 只有 API 调用才会有 429，本地模型通常是其他错误
                    if "429" in error_str:
                        wait_time = 2 ** retry_count
                        print(f"   ⚠️ 限流等待 {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"   ❌ 第 {i} 条嵌入失败 (Fatal): {e}")
                        break
            
        if not text_embeddings:
            return "保存失败: 所有数据向量化均失败，请检查模型配置。"

        print(f"✅ 向量计算完成 ({len(text_embeddings)} 条)，准备存入 Milvus...")

        # 3. 存入 Milvus
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            auto_id=True,
            drop_old=True  # 强制重建表以适应新模型的维度
        )
        
        # 存入数据
        vector_store.add_embeddings(
            texts=texts,
            embeddings=text_embeddings,
            metadatas=metadatas
        )
        
        # 强制刷新
        try:
            if hasattr(vector_store, 'col') and vector_store.col:
                vector_store.col.flush()
            elif hasattr(vector_store, 'collection') and vector_store.collection:
                vector_store.collection.flush()
        except:
            pass
        
        return f"成功将 {len(text_embeddings)} 条数据存入知识库 (Collection: {COLLECTION_NAME})"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"向量数据库操作失败: {str(e)}"
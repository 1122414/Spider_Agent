import os
import time
import json
import random
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# LangChain & Milvus
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus

# 引入 registry 以获取缓存数据
from agent.tools.registry import tool_registry

from config import *

load_dotenv()

# ========================== 配置区域 ==========================
# MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
# COLLECTION_NAME = "spider_knowledge_base"

# # Embedding 配置
# EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
# OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
# OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
# OPENAI_OLLAMA_BASE_URL = os.environ.get("MODA_OLLAMA_BASE_URL", OPENAI_BASE_URL)

def get_embedding_model():
    """
    工厂函数：自动选择 OpenAI 或 Ollama 嵌入模型
    """
    if OPENAI_OLLAMA_BASE_URL and "11434" in OPENAI_OLLAMA_BASE_URL:
        print(f"🔌 [RAG] 切换至 Ollama Embeddings (Model: {EMBEDDING_MODEL})...")
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/api/generate", "").replace("/v1", "").rstrip("/")
        return OllamaEmbeddings(base_url=base_url, model=OPENAI_OLLAMA_EMBEDDING_MODEL)
    else:
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

def _extract_items_from_structure(data: Any) -> List[Dict]:
    """
    【核心逻辑】递归查找嵌套字典中包含数据的列表
    解决类似 target_content -> items 这种深层嵌套问题
    """
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        # 1. 优先查找常见的数据容器 Key
        priority_keys = ["items", "data", "list", "target_content", "results", "products"]
        
        for key in priority_keys:
            if key in data:
                val = data[key]
                # 如果找到列表，直接返回
                if isinstance(val, list) and len(val) > 0:
                    return val
                # 如果是字典（如 target_content），递归进去找
                if isinstance(val, dict):
                    deep_items = _extract_items_from_structure(val)
                    if deep_items: 
                        return deep_items
        
        # 2. 如果常见 Key 没找到，就把 Dict 本身当做单条数据
        # 但要排除那种只包含 meta 信息（如 code: 200）的 dict
        if len(data.keys()) > 1: # 至少有点内容的
            return [data]
            
    return []

def _flatten_data_to_documents(data: Union[List, Dict]) -> List[Document]:
    """
    通用版数据扁平化：自适应各种字段名
    """
    # 1. 智能提取列表数据
    items = _extract_items_from_structure(data)
        
    if not items:
        print("⚠️ [Flatten] 未找到有效列表数据。")
        return []

    documents = []
    
    for item in items:
        if not isinstance(item, dict): continue
        
        # --- A. 动态识别 Title 和 URL ---
        title = "未命名条目"
        url = ""
        
        # 启发式关键词
        title_keywords = ["title", "name", "名称", "名", "标题", "product", "movie"]
        url_keywords = ["url", "link", "href", "链接", "跳转"]

        # 遍历所有字段来猜测 Title 和 URL
        for k, v in item.items():
            if not isinstance(v, str): continue
            k_lower = k.lower()
            
            # 猜 URL
            if not url and any(kw in k_lower for kw in url_keywords) and (v.startswith("http") or v.startswith("/")):
                url = v
            
            # 猜 Title (优先级：如果 key 包含 keyword，且 value 不像 url)
            if title == "未命名条目" and any(kw in k_lower for kw in title_keywords) and len(v) < 100:
                title = v

        # 如果没猜到 Title，尝试用第一个非 URL 的短字符串作为 Title
        if title == "未命名条目":
            for k, v in item.items():
                if isinstance(v, str) and len(v) > 2 and len(v) < 50 and not v.startswith("http"):
                    title = v
                    break

        # --- B. 构建全字段文本 (Flatten All Fields) ---
        content_parts = []
        
        # 第一层字段
        for k, v in item.items():
            # 跳过特殊字段和空值
            if k in ["children", "target_content", "items"] or v is None: 
                continue
            
            # 转换为字符串
            val_str = str(v).strip()
            if not val_str: continue
            
            # 格式化: "商品名: 洋奢发热保暖..."
            content_parts.append(f"{k}: {val_str}")
        
        parent_text = "\n".join(content_parts)
        
        if parent_text and len(parent_text) > 5:
            doc = Document(
                page_content=parent_text,
                metadata={"source": url, "title": title, "type": "parent_info"}
            )
            documents.append(doc)
            
        # --- C. 递归处理 Children (如有) ---
        children = item.get("children", [])
        # 有些网站可能把子列表叫 sub_items 等，这里可以扩展
        
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict): continue
                
                child_parts = []
                for k, v in child.items():
                    val_str = str(v).strip()
                    if val_str:
                        child_parts.append(f"{k}: {val_str}")
                
                if child_parts:
                    # 将父级 title 拼接到子级内容中，保持上下文
                    child_text = f"《{title}》的详细信息:\n" + "\n".join(child_parts)
                    child_doc = Document(
                        page_content=child_text,
                        metadata={"source": url, "title": title, "type": "child_detail"}
                    )
                    documents.append(child_doc)
                    
    return documents

def save_to_milvus(data: Union[Dict, List] = None) -> str:
    """
    将数据存入 Milvus 向量知识库 (稳健版)
    """
    actual_data = _resolve_data(data)
    if not actual_data:
        return "保存失败: 没有有效数据"

    # 转换数据
    docs = _flatten_data_to_documents(actual_data)
    
    # 过滤空文档
    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    
    if not valid_docs:
        return f"保存失败: 数据转换后为空 (原始数据类型: {type(actual_data)})"
    
    print(f"🔄 准备处理 {len(valid_docs)} 条数据片段...")

    try:
        embeddings = get_embedding_model()

        # 手动计算向量 (防止 429)
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
                    vector = embeddings.embed_query(clean_text)
                    
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)
                    text_embeddings.append(vector)
                    
                    if (i + 1) % 5 == 0:
                        print(f"   -> 已向量化 {i + 1}/{len(valid_docs)} 条")
                    
                    time.sleep(0.05)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str:
                        wait_time = 2 ** retry_count
                        print(f"   ⚠️ 限流等待 {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"   ❌ 第 {i} 条嵌入失败 (Fatal): {e}")
                        break
            
        if not text_embeddings:
            return "保存失败: 所有数据向量化均失败。"

        print(f"✅ 向量计算完成 ({len(text_embeddings)} 条)，准备存入 Milvus...")

        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            auto_id=True,
            drop_old=True 
        )
        
        vector_store.add_embeddings(
            texts=texts,
            embeddings=text_embeddings,
            metadatas=metadatas
        )
        
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
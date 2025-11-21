import os
import json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# LangChain & Milvus
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

# 引入 registry 以获取缓存数据
from agent.tools.registry import tool_registry

load_dotenv()

# 配置
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "spider_knowledge_base"
EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")

def _resolve_data(data: Union[Dict, List, None]) -> Union[Dict, List, None]:
    """解析数据，优先使用缓存"""
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
    """将树状结构的爬虫数据扁平化为 Document 对象列表"""
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
        
        # 1. 提取元数据
        title = item.get("电影名称") or item.get("title") or item.get("name") or "未知标题"
        url = item.get("链接") or item.get("url") or item.get("link") or ""
        
        # 2. 构建父级文本
        content_parts = []
        for k, v in item.items():
            # 过滤掉空值和非文本值
            if k not in ["children", "url", "link", "href", "跳转链接"] and v and isinstance(v, str) and v.strip():
                content_parts.append(f"{k}: {v.strip()}")
        
        parent_text = "\n".join(content_parts)
        
        # 严格过滤空文本
        if parent_text and len(parent_text.strip()) > 5:
            doc = Document(
                page_content=parent_text,
                metadata={"source": url, "title": title, "type": "parent_info"}
            )
            documents.append(doc)
            
        # 3. 处理 Children
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
    将数据存入 Milvus 向量知识库 (Docker版)
    """
    actual_data = _resolve_data(data)
    if not actual_data:
        return "保存失败: 没有有效数据"

    docs = _flatten_data_to_documents(actual_data)
    
    # 最后一道防线：确保没有空 Document
    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    
    if not valid_docs:
        return "保存失败: 数据转换后为空，无法入库"
    
    print(f"🔄 准备将 {len(valid_docs)} 条数据片段存入 Milvus ({MILVUS_URI})...")

    try:
        # 初始化 Embedding
        # 移除 chunk_size 参数，使用默认设置，避免触发 IndexError
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL
        )

        # 连接 Milvus
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            auto_id=True,
            drop_old=False
        )
        
        # 【核心修复】手动分批写入
        # 规避 langchain-openai 在处理大列表时的 IndexError bug
        # 每次只写 50 条，稳健性极高
        BATCH_SIZE = 50
        total_batches = (len(valid_docs) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"📦 开始分批写入 (Batch Size: {BATCH_SIZE}, Total Batches: {total_batches})...")
        
        for i in range(0, len(valid_docs), BATCH_SIZE):
            batch = valid_docs[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)
            print(f"   -> Batch {i // BATCH_SIZE + 1}/{total_batches} 完成 ({len(batch)} docs)")
        
        print(f"💾 全部完成！成功将 {len(valid_docs)} 个知识片段存入 Milvus。")
        return f"成功将 {len(valid_docs)} 条数据存入知识库 (Milvus Collection: {COLLECTION_NAME})"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"向量数据库操作失败: {str(e)}"
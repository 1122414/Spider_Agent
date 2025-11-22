import os
import time
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
        
        title = item.get("电影名称") or item.get("title") or item.get("name") or "未知标题"
        url = item.get("链接") or item.get("url") or item.get("link") or ""
        
        # 构建父级文本
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
            
        # 处理 Children
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
    将数据存入 Milvus 向量知识库 (稳健版)
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
        # 1. 初始化 Embedding 模型
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL
        )

        # 2. 【手动计算向量】
        # 绕过 LangChain 的批量处理 Bug，并增加速率限制防止 429
        text_embeddings = []
        metadatas = []
        texts = []
        
        print(f"⚡ 开始计算向量 (Manual Embedding Mode)...")
        for i, doc in enumerate(valid_docs):
            try:
                # 每次只算一条，最稳健
                # replace newlines 是官方推荐做法
                clean_text = doc.page_content.replace("\n", " ")
                vector = embeddings.embed_query(clean_text)
                
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                text_embeddings.append(vector)
                
                # 简单的进度条
                if (i + 1) % 5 == 0:
                    print(f"   -> 已向量化 {i + 1}/{len(valid_docs)} 条")
                
                # 【关键】防 429 限流：每条间隔 0.2 秒
                time.sleep(0.2)
                
            except Exception as e:
                print(f"   ⚠️ 第 {i} 条嵌入失败: {e}")
                continue

        if not text_embeddings:
            return "保存失败: 所有数据向量化均失败，请检查 API Key 或网络。"

        print(f"✅ 向量计算完成，准备存入 Milvus ({len(text_embeddings)} 条)...")

        # 3. 存入 Milvus
        # drop_old=True: 强制删除旧表，解决维度冲突 (4096 vs 1536)
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            auto_id=True,
            drop_old=True  # 【关键】强制重建表
        )
        
        # 使用 add_embeddings 直接存入算好的向量，不再让 LangChain 重新算
        vector_store.add_embeddings(
            texts=texts,
            embeddings=text_embeddings,
            metadatas=metadatas
        )
        
        print(f"💾 全部完成！成功存入 Milvus。")
        return f"成功将 {len(text_embeddings)} 条数据存入知识库 (Collection: {COLLECTION_NAME}, Recreated)"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"向量数据库操作失败: {str(e)}"
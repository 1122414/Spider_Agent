import os
from dotenv import load_dotenv
from typing import List, Dict
from utils.text_splitter import semantic_chunk_texts
from rag.vectorstore import get_chroma_client_and_store
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
MODA_OPENAI_EMBEDDING_API_KEY = os.environ.get("MODA_OPENAI_EMBEDDING_API_KEY")
MODA_OPENAI_EMBEDDING_BASE_URL = os.environ.get("MODA_OPENAI_EMBEDDING_BASE_URL")
COLLECTION = os.environ.get("COLLECTION_NAME", "auto_crawler_collection")

def ingest_crawled(doc_meta: Dict):
    """
    处理从爬虫获取的文档数据并存入 ChromaDB
    doc_meta 结构: {"url":..., "html":..., "title":..., "target_content": [...]}
    1) 从 target_content 提取段落文本
    2) 将文本进行语义分块
    3) 转为 embedding 并写入 Chroma
    """
    # 从爬虫结果中提取文本内容
    target_content = doc_meta.get("target_content", [])
    title = doc_meta.get("title", "")
    url = doc_meta.get("url", "")

    # 提取段落文本
    paragraphs = []
    for item in target_content:
        if isinstance(item, dict):
            # 如果 target_content 是字典列表，提取文本字段
            text = item.get('text', '') or item.get('content', '') or str(item)
        else:
            text = str(item)
        
        if text and text.strip():
            paragraphs.append(text.strip())

    # 如果没有提取到段落，尝试从 HTML 中提取
    if not paragraphs:
        html_content = doc_meta.get("html", "")
        if html_content:
            # 简单的 HTML 文本提取
            import re
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            # 提取所有段落文本
            text_elements = soup.find_all(['p', 'div', 'span', 'li', 'td', 'th'])
            for element in text_elements:
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # 只保留有足够内容的文本
                    paragraphs.append(text)

    docs = []
    for p in paragraphs:
        # 对每个段落进行语义分块
        chunks = semantic_chunk_texts(p, chunk_size=800, overlap=120)
        for chunk in chunks:
            docs.append({
                "text": chunk,
                "meta": {"title": title, "source": url}
            })

    # 如果仍然没有文档，使用标题和URL作为最小文档
    if not docs:
        docs.append({
            "text": f"{title} - {url}",
            "meta": {"title": title, "source": url}
            })

    # embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=MODA_OPENAI_EMBEDDING_API_KEY, base_url=MODA_OPENAI_EMBEDDING_BASE_URL)
    # store to chroma
    chroma_store = get_chroma_client_and_store(collection_name=COLLECTION, embeddings=embeddings)
    texts = [d["text"] for d in docs]
    metadatas = [d["meta"] for d in docs]

    # 添加到 ChromaDB
    chroma_store.add_texts(texts=texts, metadatas=metadatas)
    return {"ingested_docs": len(texts)}

from typing import List, Dict
from utils.text_splitter import semantic_chunk_texts
from rag.vectorstore import get_qdrant_client_and_store
from langchain.embeddings import OpenAIEmbeddings
import os

EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION = os.environ.get("COLLECTION_NAME", "auto_crawler_collection")

def ingest_crawled(doc_meta: Dict):
    """
    doc_meta: {"url":..., "title":..., "paragraphs":[...]}
    1) 将 paragraphs 合并或分别做 semantic chunk
    2) 转为 embedding 并写入 Qdrant
    """
    paragraphs = doc_meta.get("paragraphs", [])
    title = doc_meta.get("title", "")
    url = doc_meta.get("url", "")

    docs = []
    for p in paragraphs:
        # for each paragraph, produce chunks
        chunks = semantic_chunk_texts(p, chunk_size=800, overlap=120)
        for chunk in chunks:
            docs.append({
                "text": chunk,
                "meta": {"title": title, "source": url}
            })

    # embeddings
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    # store to qdrant
    qdrant_store = get_qdrant_client_and_store(collection_name=COLLECTION, embeddings=embeddings)
    texts = [d["text"] for d in docs]
    metadatas = [d["meta"] for d in docs]

    qdrant_store.add_texts(texts=texts, metadatas=metadatas)
    return {"ingested_docs": len(texts)}

from chromadb import Client
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import os

CHROMA_URL = os.environ.get("CHROMA_URL", "http://localhost:8000")
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY", None)

def get_chroma_client():
    client = Client(Settings(
        chroma_server_host=CHROMA_URL.split(":")[1].strip("//"),
        chroma_server_http_port=int(CHROMA_URL.split(":")[-1]),
        chroma_server_ssl=False,
        chroma_server_api_key=CHROMA_API_KEY
    ))
    return client

def get_chroma_client_and_store(collection_name: str, embeddings: Embeddings):
    client = get_chroma_client()
    # create collection if not exists
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        # create collection with default vector size (you must set correct vector size by embedding model)
        # We'll attempt to infer vector_size using the embedding provider (LangChain wrapper) -> not always available
        vector_size = 1536
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    # wrap with LangChain Chroma
    chroma_store = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
    return chroma_store

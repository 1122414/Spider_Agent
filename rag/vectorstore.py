import os
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

VECTOR_DB_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = os.environ.get("VECTOR_DB_PORT", 7070)

def get_chroma_client():
    client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
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

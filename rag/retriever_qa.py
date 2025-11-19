import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from rag.vectorstore import get_chroma_client_and_store

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION = os.environ.get("COLLECTION_NAME", "auto_crawler_collection")

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
llm = ChatOpenAI(model=MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

# prepare retriever for queries
def get_retriever(k:int = 6, use_mmr:bool=True):
    store = get_chroma_client_and_store(collection_name=COLLECTION, embeddings=embeddings)
    retriever = store.as_retriever(search_type="mmr" if use_mmr else "similarity",
                                   search_kwargs={"k": k})
    return retriever

def qa_interaction(query: str, k:int = 6):
    retriever = get_retriever(k=k)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="map_reduce")
    answer = qa.run(query)
    return answer

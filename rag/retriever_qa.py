import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 配置
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "spider_knowledge_base"
EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
MODEL_NAME = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def qa_interaction(question: str) -> str:
    print(f"🤔 RAG Searching for: {question}")
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )

    try:
        # 连接 Milvus (Docker)
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )
        
        # 检索 Top 3
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        template = """你是一个基于本地知识库的智能助手。请根据下面的【上下文】内容回答用户的问题。
        如果上下文中没有相关信息，请诚实地说“知识库中未找到相关信息”。

        【上下文】:
        {context}

        【问题】:
        {question}

        【回答】:"""
        
        custom_rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(question)
        return result

    except Exception as e:
        return f"RAG 系统出错: {str(e)}"
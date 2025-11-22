import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from agent.prompt_template import RAG_PROMPT

load_dotenv()

# 配置
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "spider_knowledge_base"
EMBEDDING_MODEL = os.environ.get("MODA_EMBEDDING_MODEL", "text-embedding-3-small")
MODEL_NAME = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")

def format_docs(docs):
    # 在合并文档时，给每个片段加个序号，方便模型引用
    return "\n\n".join(f"[片段 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

def determine_search_kwargs(question: str) -> dict:
    """
    【核心逻辑】根据问题类型，动态调整检索策略 (Dynamic K)
    """
    # 定义触发“全局检索”的关键词
    global_keywords = ["全部", "所有", "列表", "清单", "总结", "分析", "概括", "all", "summary", "list", "有哪些", "统计"]
    
    # 检查问题是否包含关键词
    is_global_query = any(kw in question.lower() for kw in global_keywords)
    
    if is_global_query:
        print("🚀 检测到全局/总结性提问，启动【强力检索模式】(k=100)...")
        # GPT-4o-mini 上下文很大(128k)，可以轻松处理 100 个片段 (约 2w token)
        # 这样就能一次性把几十部电影的信息都喂给模型，让它做总结
        return {"k": 100} 
    else:
        print("🔍 检测到具体事实提问，使用【精准检索模式】(k=10)...")
        return {"k": 10}

def qa_interaction(question: str) -> str:
    print(f"🤔 RAG Searching for: {question}")
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1, # 总结类任务降低温度，减少编造
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )

    try:
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
        )
        
        # 1. 动态决定 k 值
        # 如果你问“有哪些电影”，这里就会让 Milvus 返回最相关的 100 条数据
        # 基本上就能覆盖你爬取的所有电影了
        search_kwargs = determine_search_kwargs(question)
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        # 2. 增强版 Prompt
        # 明确告诉模型它将收到大量数据，需要进行综合处理
        template = RAG_PROMPT
        
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
import os
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 导入核心组件
from agent.tools.registry import tool_registry
from agent.decision_engine import init_decision_engine

# 2. 导入工具函数
# 爬虫工具
from agent.tools.crawl_tool import sync_playwright_fetch, sync_hierarchical_crawl
# 保存工具 (文件/数据库)
from agent.tools.save_tool import save_to_csv, save_to_json, save_to_postgres
# RAG 入库工具
from agent.tools.ingest_tool import save_to_milvus

# 加载环境变量
load_dotenv()

MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
MODEL = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")

def setup_system():
    """系统初始化与装配"""
    print(">>> 系统初始化中...")
    
    # A. 初始化 LLM
    chat = ChatOpenAI(
        model=MODEL, 
        temperature=0, 
        openai_api_key=MODA_OPENAI_API_KEY, 
        openai_api_base=MODA_OPENAI_BASE_URL
    )
    
    # B. 注册工具箱
    
    # --- 1. 爬虫类工具 ---
    tool_registry.register_tool(
        tool_name="web_crawler",
        description="基础爬虫：提取单页面信息。参数: url, target (字段列表)。",
        func=sync_playwright_fetch
    )
    
    tool_registry.register_tool(
        tool_name="hierarchical_crawler",
        description="""
        多层级深度爬虫：支持从列表页->详情页->更多详情页的递归抓取。
        参数: 
        - url: 起始网址。
        - crawl_scopes: 一个二维数组，定义每一层的抓取目标。
          例如抓取3层: [ ["动漫名", "链接"], ["播放线路链接"], ["评论内容", "点赞"] ]
        - max_items: (可选) 每一页递归抓取的最大条目数，默认100（即抓取整页）。
        - max_pages: (可选) 每一层列表页的最大翻页数，默认10。
        """,
        func=sync_hierarchical_crawl
    )
    
    # --- 2. 基础保存工具 ---
    tool_registry.register_tool(
        tool_name="save_to_json",
        description="""
        将数据保存为 JSON 文件。
        参数:
        - data: (可选) 要保存的数据。不传则自动使用上一步爬取的数据。
        - filename_prefix: (可选) 文件名前缀。
        """,
        func=save_to_json
    )
    
    tool_registry.register_tool(
        tool_name="save_to_csv",
        description="""
        将数据保存为 CSV 表格文件。会自动处理嵌套结构。
        参数:
        - data: (可选) 要保存的数据。不传则自动使用上一步爬取的数据。
        - filename_prefix: (可选) 文件名前缀。
        """,
        func=save_to_csv
    )
    
    tool_registry.register_tool(
        tool_name="save_to_postdb",
        description="""
        将数据保存到 PostgreSQL 数据库。
        参数:
        - data: (可选) 要保存的数据。不传则自动使用上一步爬取的数据。
        - table_name: (可选) 数据库表名，默认为 'crawled_data'。
        注意：环境必须配置 POSTGRES_CONNECTION_STRING。
        """,
        func=save_to_postgres
    )

    # --- 3. RAG 知识库工具 ---
    tool_registry.register_tool(
        tool_name="save_to_knowledge_base",
        description="""
        将爬取的数据存入 Milvus 向量知识库，以便后续进行 RAG 问答。
        无需参数，自动使用上一步爬取的数据。
        """,
        func=save_to_milvus
    )
    
    # C. 初始化决策引擎
    engine = init_decision_engine(chat)
    print(">>> 系统初始化完成。")
    return engine

def interactive_agent_loop(decision_engine):
    """Agent 交互主循环"""
    print("\n🤖 AutoCrawlerAgent V2 就绪 — 输入自然语言任务（输入 exit 退出）")
    print("💡 提示：输入 'qa <问题>' 可直接针对知识库提问。")
    
    while True:
        try:
            user_input = input("\n👤 User > ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("👋 Bye!")
                break
            
            if not user_input.strip():
                continue

            # --- 特殊指令：直接进入 RAG 问答模式 ---
            # 如果用户开头输入 "qa" 或 "ask"，直接调用 RAG，不走 Agent 决策
            if user_input.lower().startswith("qa ") or user_input.lower().startswith("ask "):
                query = user_input.split(" ", 1)[1]
                try:
                    # 延迟导入，确保环境准备好
                    from rag.retriever_qa import qa_interaction
                    qa_result = qa_interaction(query)
                    print(f"\n🤖 [Knowledge Base]: {qa_result}")
                except Exception as e:
                    print(f"⚠️ RAG Error: {e}")
                    print("提示: 请确保已安装 pymilvus, langchain-milvus 并正确配置了 Milvus 服务。")
                continue

            # --- 正常 Agent 流程 ---
            print("🤖 Agent正在思考并执行任务...")
            
            # 1. Agent 决策与执行
            result = decision_engine.think_and_act(user_input)
            
            print("\n✅ 任务执行结果：")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 2. 引导提示
            if result.get("status") == "completed":
                print("\n💡 提示: 如果你已将数据存入知识库，可以直接输入 'qa <问题>' 进行提问。")

        except KeyboardInterrupt:
            print("\n操作已取消")
            break
        except Exception as e:
            print(f"\n❌ 发生未捕获异常: {e}")

if __name__ == "__main__":
    # 1. 装配系统
    engine = setup_system()
    
    # 2. 启动循环
    interactive_agent_loop(engine)
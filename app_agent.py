import os
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 导入核心组件
from agent.tools.registry import tool_registry
from agent.decision_engine import init_decision_engine

# 2. 导入具体的工具函数 (在这里导入，而不是在 registry 中)
from agent.tools.crawl_tool import sync_playwright_fetch, sync_hierarchical_crawl
from agent.tools.save_tool import save_to_csv, save_to_json, save_to_postgres

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
    
    # B. 【关键步骤】在这里注册工具
    
    # --- 1. 注册爬虫工具 ---
    tool_registry.register_tool(
        tool_name="web_crawler",
        description="基础爬虫：提取单页面信息。参数: url, target (字段列表), max_scrolls(默认0)。",
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
        - max_items: 每一层递归抓取的最大条目数
        - max_pages: 每一层列表页的最大翻页数
        """,
        func=sync_hierarchical_crawl
    )
    
    # --- 2. 注册保存工具 ---
    
    # JSON 保存
    tool_registry.register_tool(
        tool_name="save_to_json",
        description="""
        将数据保存为 JSON 文件。
        参数:
        - data: 要保存的数据对象 (通常是上一步爬虫的结果)。
        - filename_prefix: (可选) 文件名前缀，如 'anime_data'。
        """,
        func=save_to_json
    )
    
    # CSV 保存
    tool_registry.register_tool(
        tool_name="save_to_csv",
        description="""
        将数据保存为 CSV 表格文件。会自动处理嵌套结构。
        参数:
        - data: 要保存的数据对象。
        - filename_prefix: (可选) 文件名前缀。
        """,
        func=save_to_csv
    )
    
    # 数据库保存
    tool_registry.register_tool(
        tool_name="save_to_db",
        description="""
        将数据保存到 PostgreSQL 数据库。
        参数:
        - data: 要保存的数据对象。
        - table_name: (可选) 数据库表名，默认为 'crawled_data'。
        注意：环境必须配置 POSTGRES_CONNECTION_STRING。
        """,
        func=save_to_postgres
    )
    
    # C. 初始化决策引擎
    engine = init_decision_engine(chat)
    print(">>> 系统初始化完成。")
    return engine

def interactive_agent_loop(decision_engine):
    """Agent 交互主循环"""
    print("\n🤖 AutoCrawlerAgent V2 就绪 — 输入自然语言任务（输入 exit 退出）")
    
    while True:
        try:
            user_input = input("\n👤 User > ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("👋 Bye!")
                break
            
            if not user_input.strip():
                continue

            print("🤖 Agent正在思考并执行任务...")
            
            # --- 1. Agent 决策与执行 ---
            result = decision_engine.think_and_act(user_input)
            
            print("\n✅ 任务执行结果：")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # --- 2. (可选) 进入知识库问答模式 ---
            # 逻辑修正：如果结果成功，或者是用户显式要求搜索/问答
            if "knowledge" in user_input.lower() or "search" in user_input.lower():
                print("\n📚 进入知识库问答模式（输入 new 返回主菜单，exit 退出程序）")
                
                while True:
                    q = input("\n(RAG) Q > ")
                    
                    if q.strip().lower() in ("new", "back"):
                        break
                    if q.strip().lower() in ("exit", "quit"):
                        return # 彻底退出程序
                        
                    # 延迟导入，避免如果没有 RAG 模块导致整个程序跑不起来
                    try:
                        from rag.retriever_qa import qa_interaction
                        qa_result = qa_interaction(q)
                        print(f"\n(RAG) A > {qa_result}")
                    except ImportError:
                        print("⚠️ 未找到 rag.retriever_qa 模块，跳过问答。")
                        break
                    except Exception as e:
                        print(f"⚠️ RAG 运行时错误: {e}")

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
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from agent.tools.crawl_tool import sync_playwright_fetch
from rag.retriever_qa import qa_interaction
from agent.decision_engine import init_decision_engine


load_dotenv()

MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
MODEL = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")
COLLECTION = os.environ.get("COLLECTION_NAME", "auto_crawler_collection")


# 初始化 LLM
chat = ChatOpenAI(
    model=MODEL, 
    temperature=0, 
    openai_api_key=MODA_OPENAI_API_KEY, 
    openai_api_base=MODA_OPENAI_BASE_URL
)


class AutoCrawlerAgentV2:
    """使用LangChain官方Agent框架和决策引擎的完整实现"""
    
    def __init__(self):
        self.decision_engine = init_decision_engine(chat)
        
    def run_agent_task(self, task: str):
        """使用决策引擎执行任务 - 真正的Agent模式"""
        
        print(f"Agent V2 正在处理任务: {task}")
        
        # 使用决策引擎进行ReAct决策循环
        result = self.decision_engine.think_and_act(task)
        
        return result
    
    def interactive_loop(self):
        """Agent交互循环"""
        print("AutoCrawlerAgent V2 — 输入自然语言任务（exit 退出）")
        
        while True:
            user_input = input("\n> ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
                
            print("Agent正在思考并执行任务...")
            task_result = self.run_agent_task(user_input)
            
            print("\n任务执行结果：")
            print(json.dumps(task_result, ensure_ascii=False, indent=2))
            
            # 检查是否需要知识库问答
            if "knowledge" in user_input.lower() or "search" in user_input.lower():
                print("\n你可以继续提问或输入new开始新任务）")
            
            while True:
                q = input("\nQ> ")
                if q.strip().lower() in ("new", "back"):
                  break
                    
                if q.strip().lower() in ("exit","quit"):
                  return
                    
                qa_result = qa_interaction(q)
                print("\nA> ", qa_result)


# 全局Agent实例
advanced_agent = AutoCrawlerAgentV2()

# def init_advanced_agent():
#     """初始化高级Agent"""
#     global advanced_agent
#     advanced_agent = AutoCrawlerAgentV2()

# init_advanced_agent()

# if __name__ == "__main__":
#     init_advanced_agent()
#     advanced_agent.interactive_loop()
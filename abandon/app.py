import os
import json
from dotenv import load_dotenv
from abandon.langchain_agent import run_agent_task
from rag.retriever_qa import qa_interaction

load_dotenv()

def interactive_loop():
    print("AutoCrawlerAgent — 输入自然语言任务（exit 退出）")
    while True:
        user = input("\n> ")
        if user.strip().lower() in ("exit", "quit"):
            break
        # run parsing + planning + crawl + ingest to vector DB
        print("解析并执行任务，可能需要一点时间...")
        task_result = run_agent_task(user)
        print("\n任务结果摘要：")
        print(json.dumps(task_result.get("summary", {}), ensure_ascii=False, indent=2))
        print("\n你可以现在对已导入的知识进行问答，输入问题（或直接输入 new 来新任务）:")
        while True:
            q = input("\nQ> ")
            if q.strip().lower() in ("new", "back"):
                break
            if q.strip().lower() in ("exit","quit"):
                return
            ans = qa_interaction(q)
            print("\nA> ", ans)

if __name__ == "__main__":
    interactive_loop()

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from agent.prompt_template import FIND_URL_SYSTEM_PROMPT
# 修改引入：引入新的同步包装函数 sync_playwright_fetch
from agent.tools.crawl_tool import sync_playwright_fetch
from agent.tools.ingest_tool import ingest_crawled

load_dotenv()

MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
MODEL = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")

# structured parser schema
schemas = [
    ResponseSchema(name="website", description="完整的目标网址，如果没有可以留空"),
    ResponseSchema(name="platform", description="平台名（例如携程/马蜂窝/目标站点）"),
    ResponseSchema(name="keywords", description="检索关键词"),
    ResponseSchema(name="fields", description="字段列表，示例：['标题','评分','价格']")
]

# StructuredOutputParser (结构化输出解析器)
parser = StructuredOutputParser.from_response_schemas(schemas)

prompt_template = """
{system_prompt}

{format_instructions}

用户输入：
{query}
"""

# 初始化 LLM
chat = ChatOpenAI(
    model=MODEL, 
    temperature=0, 
    openai_api_key=MODA_OPENAI_API_KEY, 
    openai_api_base=MODA_OPENAI_BASE_URL
)

def parse_task(nl_text: str) -> dict:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted = {
        "system_prompt": FIND_URL_SYSTEM_PROMPT,
        "format_instructions": parser.get_format_instructions(),
        "query": nl_text
    }
    messages = prompt.format_prompt(**formatted).to_messages()

    resp = chat.invoke(messages)
    # resp.content contains JSON
    try:
        parsed = parser.parse(resp.content)
    except Exception:
        # fallback to simple attempt
        try:
            parsed = json.loads(resp.content)
        except Exception:
            parsed = {"website": "", "platform": "", "keywords": nl_text, "fields": []}
    return parsed

def run_agent_task(nl_text: str):
    """
    1. parse
    2. try to find a target url (if not present you may implement heuristics or search)
    3. call sync_playwright_fetch (Wrapper)
    4. ingest
    """
    print(f"正在解析任务: {nl_text}")
    task = parse_task(nl_text)
    
    url = task.get("website")
    target = task.get("fields")
    
    if not url:
        # TODO: implement site lookup by platform+keywords (search engine)
        return {"error": "未识别到明确 URL，请在输入中包含完整URL或后续实现site-search", "task": task}
    
    print(f"开始抓取 URL: {url}")
    
    # [关键修改] 使用同步包装函数，不再直接调用 asyncio.run()
    # 这一步会自动处理 Event Loop 的问题
    fetched = sync_playwright_fetch(url, target)
    
    # 4. ingest
    ingest_info = ingest_crawled(fetched)
    
    return {
        "task": task, 
        "summary": {
            "url": url, 
            "title": fetched.get("title"), 
            **ingest_info
        }
    }
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agent.prompt_template import SYSTEM_PROMPT
from agent.tools.crawl_tool import drission_fetch
from agent.tools.ingest_tool import ingest_crawled
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# structured parser schema
schemas = [
    ResponseSchema(name="website", description="完整的目标网址，如果没有可以留空"),
    ResponseSchema(name="platform", description="平台名（例如携程/马蜂窝/目标站点）"),
    ResponseSchema(name="keywords", description="检索关键词"),
    ResponseSchema(name="fields", description="字段列表，示例：['标题','评分','价格']")
]
parser = StructuredOutputParser.from_response_schemas(schemas)

prompt_template = """
{system_prompt}

{format_instructions}

用户输入：
{query}
"""

chat = ChatOpenAI(model=MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

def parse_task(nl_text: str) -> dict:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted = {
        "system_prompt": SYSTEM_PROMPT,
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
    3. call drission_fetch
    4. ingest
    """
    task = parse_task(nl_text)
    url = task.get("website")
    if not url:
        # TODO: implement site lookup by platform+keywords (search engine)
        return {"error": "未识别到明确 URL，请在输入中包含完整URL或后续实现site-search"}
    # 3. fetch
    fetched = drission_fetch(url)
    # 4. ingest
    ingest_info = ingest_crawled(fetched)
    return {"task": task, "summary": {"url": url, "title": fetched.get("title"), **ingest_info}}

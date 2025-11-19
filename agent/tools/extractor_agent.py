import os
import re
import html
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agent.prompt_template import SCRAWL_DATA_SYSTEM_PROMPT
from utils.extractor_utils import summarize_structure, nodes_to_text_summary

load_dotenv()

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# BASE_URL = os.environ.get("OPENAI_BASE_URL")
MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
MODEL = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")
# llm = ChatOpenAI(model=MODEL, temperature=0, openai_api_key=MODA_OPENAI_API_KEY, openai_api_base=MODA_OPENAI_BASE_URL)
# resp = llm.invoke("你是谁？")
# print(resp)  # 测试连接


class ExtractorAgent:
    def __init__(self):
        # self.llm = ChatOpenAI(model=MODEL, temperature=0, openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL)
        self.llm = ChatOpenAI(model=MODEL, temperature=0, openai_api_key=MODA_OPENAI_API_KEY, openai_api_base=MODA_OPENAI_BASE_URL)

    def get_content(self, fetched_html, target: List[str],source) -> Dict:
        # 1. 得到 html (drission_fetch 返回的 html)
        html = fetched_html  # 你的页面 html 字符串

        # 2. 生成节点摘要（清晰、无转义）
        # 使用了html2text.transform_documents之后就没必要了
        # nodes = summarize_structure(html, max_nodes=500, min_text_len=8)
        # summary_text = nodes_to_text_summary(nodes, max_lines=200)

        # 3. 发送给 LLM
        prompt = PromptTemplate.from_template(SCRAWL_DATA_SYSTEM_PROMPT)

        resp = self.llm.invoke(prompt.format(user_query=target, summary=html, source=source))
        # 解析 resp.content -> 索引列表
        # 尝试将返回内容解析成 JSON
        content = resp.content.strip()

        try:
            # 常规解析
            result = json.loads(content)
        except json.JSONDecodeError:
            # 如果模型输出了 markdown 格式（如 ```json ... ```），则先清洗
            cleaned = (
                content.replace("```json", "")
                      .replace("```", "")
                      .strip()
            )
            result = json.loads(cleaned)

        return result

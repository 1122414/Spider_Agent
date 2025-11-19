import os
import re
import html
import json
from typing import List, Dict, Any, Set, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agent.prompt_template import SCRAWL_DATA_SYSTEM_PROMPT

load_dotenv()

MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
MODEL = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")

class ExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL, 
            temperature=0.1, # 降低温度以提高格式稳定性
            openai_api_key=MODA_OPENAI_API_KEY, 
            openai_api_base=MODA_OPENAI_BASE_URL
        )

    def get_content(self, fetched_html: str, target: List[str], source: str) -> Any:
        """
        根据 HTML 和目标字段，使用 LLM 提取结构化数据
        """
        # 1. 预检查：如果 HTML 内容为空或过短，直接跳过
        if not fetched_html or len(fetched_html.strip()) < 10:
            print("⚠️ 警告: 输入的 HTML 内容为空或过短，跳过提取。")
            return []

        # ============================================================
        # 【核心修复】严格的长度截断机制，避免 400 Input Length Error
        # ============================================================
        # 设定安全阈值。报错提示上限是 129024，我们留出 20k 给 System Prompt 和 User Query
        # 实际上 100k 字符通常能覆盖绝大多数网页的核心内容
        MAX_INPUT_LENGTH = 256000
        
        if len(fetched_html) > MAX_INPUT_LENGTH:
            print(f"⚠️ 输入内容过长 ({len(fetched_html)} chars)，正在截断至 {MAX_INPUT_LENGTH} 字符以避免报错...")
            # 截断并添加标记，让 LLM 知道内容不完整
            fetched_html = fetched_html[:MAX_INPUT_LENGTH] + "\n\n...(Content Truncated due to length limit)..."

        # 2. 发送给 LLM
        prompt = PromptTemplate.from_template(SCRAWL_DATA_SYSTEM_PROMPT)
        
        try:
            # 将列表类型的 target 转为字符串，方便 Prompt 理解
            resp = self.llm.invoke(prompt.format(user_query=str(target), summary=fetched_html, source=source))
            content = resp.content.strip()
        except Exception as e:
            # 捕获所有 LLM 调用异常，防止整个 Agent 崩溃
            error_str = str(e)
            print(f"❌ LLM 调用失败: {error_str}")
            
            # 如果截断后依然报错（极少数情况），返回友好的错误结构
            if "400" in error_str or "length" in error_str.lower():
                return {"error": "Content too long for LLM", "details": "Please try reducing the crawl scope or target."}
            
            return {"error": "LLM invocation failed", "details": error_str}

        # 3. 解析 JSON (包含重试、正则提取和清洗逻辑)
        result = self._parse_json_safely(content)

        # 4. 结果去重 (仅针对列表结果)
        if isinstance(result, list):
            return self._deduplicate_items(result)
        
        # 如果解析出错返回了字典形式的错误信息
        if isinstance(result, dict) and "error" in result:
            print(f"⚠️ 提取失败: {result['error']}")
            # 在多层爬取中，提取失败最好返回空列表，以免打断递归
            return []

        return result

    def _parse_json_safely(self, text: str) -> Union[List, Dict]:
        """
        安全地解析 JSON，包含多层清洗策略
        """
        # 策略 1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 2: 去除 Markdown 代码块标记
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # 策略 3: 正则表达式提取 [ ... ]
        # 专门应对 LLM 在 JSON 前后加废话的情况 (例如: "Here is the json: [...]")
        try:
            # re.DOTALL 让 . 匹配换行符
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                potential_json = match.group(0)
                return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
            
        # 策略 4: 尝试提取对象 { ... } (如果返回的是单个对象而非列表)
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                potential_json = match.group(0)
                return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

        # 策略 5: 彻底失败
        print(f"❌ JSON 解析彻底失败。原始内容预览: {text[:100]}...")
        return {"error": "Failed to parse JSON", "raw_content": text}

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """
        对提取结果列表进行智能去重
        策略：自动查找 url/link 字段，如果 URL 相同则视为重复条目。
        """
        if not items:
            return []

        unique_items = []
        seen_urls = set()
        
        # 可能表示链接的键名
        target_keys = ["url", "link", "href", "链接", "详情页链接", "文章链接", "电影链接", "source", "detail_url", "full_url"]

        for item in items:
            if not isinstance(item, dict):
                unique_items.append(item)
                continue

            found_url = None
            # 智能寻找该条目中的 URL 值
            for k, v in item.items():
                if k.lower() in target_keys and v and isinstance(v, str):
                    found_url = v.strip()
                    break
            
            if found_url:
                # 标准化 URL (去除末尾斜杠)
                normalized_url = found_url.rstrip('/')
                
                if normalized_url in seen_urls:
                    continue # 跳过重复
                
                seen_urls.add(normalized_url)
                unique_items.append(item)
            else:
                # 如果没找到 URL 字段，默认保留
                unique_items.append(item)

        if len(items) != len(unique_items):
            print(f"🔍 ExtractorAgent 去重优化: {len(items)} -> {len(unique_items)} 条")
        
        return unique_items
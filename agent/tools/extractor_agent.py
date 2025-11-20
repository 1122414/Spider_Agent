import os
import re
import html
import json
import time
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
        支持自动分块处理超长文本 (Map-Reduce 模式)
        """
        # 1. 预检查
        if not fetched_html or len(fetched_html.strip()) < 10:
            print("⚠️ 警告: 输入的 HTML 内容为空或过短，跳过提取。")
            return []

        # ============================================================
        # 【核心升级】分块处理策略 (Map-Reduce)
        # ============================================================
        # 设定单块最大字符数。
        # 中文环境下，安全阈值建议设为 20,000 - 30,000 字符 (约 60k-90k bytes < 129k limit)
        # 留出余量给 Prompt
        CHUNK_SIZE = 30000
        
        # 如果内容总长度小于阈值，直接处理 (Fast Path)
        if len(fetched_html) <= CHUNK_SIZE:
            return self._process_single_chunk(fetched_html, target, source)

        # 如果内容过长，进行分块
        print(f"📦 内容过长 ({len(fetched_html)} chars)，启动分块提取模式 (Chunk Size: {CHUNK_SIZE})...")
        
        chunks = self._split_text_by_lines(fetched_html, CHUNK_SIZE)
        print(f"   -> 切分为 {len(chunks)} 个块，开始逐块提取...")

        all_results = []
        
        for i, chunk in enumerate(chunks):
            print(f"   🔄 处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} chars)...")
            
            # 提取当前块的数据
            chunk_result = self._process_single_chunk(chunk, target, source)
            
            # 收集结果
            if isinstance(chunk_result, list):
                all_results.extend(chunk_result)
                print(f"      ✅ 第 {i+1} 块提取到 {len(chunk_result)} 条数据")
            elif isinstance(chunk_result, dict) and "error" not in chunk_result:
                 # 如果 LLM 返回了单个对象，包一层放进去
                all_results.append(chunk_result)
            
            # (可选) 简单的速率限制，防止并发太快被 API 封禁
            # time.sleep(0.5) 

        print(f"📦 分块提取完成，原始总条数: {len(all_results)}")

        # 4. 全局合并后去重 (Reduce & Deduplicate)
        final_results = self._deduplicate_items(all_results)
        return final_results

    def _process_single_chunk(self, chunk_text: str, target: List[str], source: str) -> Any:
        """
        内部方法：处理单个文本块的提取
        """
        prompt = PromptTemplate.from_template(SCRAWL_DATA_SYSTEM_PROMPT)
        
        try:
            resp = self.llm.invoke(prompt.format(user_query=str(target), summary=chunk_text, source=source))
            content = resp.content.strip()
        except Exception as e:
            error_str = str(e)
            print(f"❌ LLM 调用失败 (Chunk): {error_str[:100]}...")
            return [] # 单块失败不影响整体，返回空列表

        # 解析 JSON
        result = self._parse_json_safely(content)

        # 简单的错误检查
        if isinstance(result, dict) and "error" in result:
            # print(f"⚠️ 块提取解析失败: {result['error']}")
            return []
            
        return result

    def _split_text_by_lines(self, text: str, max_length: int) -> List[str]:
        """
        按行安全切分文本，确保不切断完整的行（Markdown结构）
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line) + 1 # +1 是换行符
            
            # 如果单行本身就超长（极端情况），强制切断或单独成块
            if line_len > max_length:
                # 如果当前块有内容，先保存
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # 超长行单独作为一块（或者你可以选择强制截断，这里选择保留）
                chunks.append(line[:max_length]) 
                continue

            # 如果加入这行会超长，则保存当前块，开启新块
            if current_length + line_len > max_length:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        # 保存最后一块
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

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
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
        # 策略 4: 尝试提取对象 { ... }
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

        # 策略 5: 彻底失败
        # print(f"❌ JSON 解析彻底失败。原始内容预览: {text[:50]}...")
        return {"error": "Failed to parse JSON", "raw_content": text}

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """
        对提取结果列表进行智能去重
        """
        if not items:
            return []

        unique_items = []
        seen_urls = set()
        
        target_keys = ["url", "link", "href", "链接", "详情页链接", "文章链接", "电影链接", "source", "detail_url", "full_url"]

        for item in items:
            if not isinstance(item, dict):
                unique_items.append(item)
                continue

            found_url = None
            for k, v in item.items():
                if k.lower() in target_keys and v and isinstance(v, str):
                    found_url = v.strip()
                    break
            
            if found_url:
                normalized_url = found_url.rstrip('/')
                if normalized_url in seen_urls:
                    continue 
                seen_urls.add(normalized_url)
                unique_items.append(item)
            else:
                unique_items.append(item)

        if len(items) != len(unique_items):
            print(f"🔍 ExtractorAgent 全局去重: {len(items)} -> {len(unique_items)} 条")
        
        return unique_items
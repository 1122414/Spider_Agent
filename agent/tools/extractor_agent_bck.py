import os
import re
import html
import json
from typing import List, Dict, Any, Set, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import Html2TextTransformer

from agent.tools.dom_helper import dom_analyzer
from agent.prompt_template import XPATH_ANALYSIS_PROMPT
from agent.prompt_template import SCRAWL_DATA_SYSTEM_PROMPT

from config import *

load_dotenv()

class ExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME, 
            temperature=0.1, # 降低温度以提高格式稳定性
            openai_api_key=OPENAI_API_KEY, 
            openai_api_base=OPENAI_BASE_URL
        )

    def get_content(self, fetched_html: str, target: List[str], source: str) -> Dict[str, Any]:
        """
        根据 HTML 和目标字段，使用 LLM 提取结构化数据
        Return: {"items": List[Dict], "next_page_url": str | None}
        """
        # 1. 预检查
        if not fetched_html or len(fetched_html.strip()) < 10:
            print("⚠️ 警告: 输入的 HTML 内容为空或过短，跳过提取。")
            return {"items": [], "next_page_url": None}

        # ============================================================
        # 分块策略 (Map-Reduce)
        # ============================================================
        CHUNK_SIZE = 20000  # 20k 字符安全阈值
        
        # Fast Path: 不分块
        if len(fetched_html) <= CHUNK_SIZE:
            return self._process_single_chunk(fetched_html, target, source)

        # Slow Path: 分块处理
        print(f"📦 内容过长 ({len(fetched_html)} chars)，启动分块提取 (Chunk Size: {CHUNK_SIZE})...")
        chunks = self._split_text_by_lines(fetched_html, CHUNK_SIZE)
        print(f"   -> 切分为 {len(chunks)} 块，开始逐块提取...")

        all_items = []
        detected_next_page = None
        
        for i, chunk in enumerate(chunks):
            # 提取当前块
            chunk_result = self._process_single_chunk(chunk, target, source)
            
            # 1. 收集 items
            items = chunk_result.get("items", [])
            if items:
                all_items.extend(items)
                print(f"✅ 第 {i+1} 块提取到 {len(items)} 条数据")
            
            # 2. 收集 next_page_url 
            # 翻页链接通常在页面的底部（即最后几个块中），但也可能在中间（如“更多”按钮）
            # 策略：只要发现有效翻页链接，就记录下来。后续块如果发现新的，可以覆盖（假设底部的是真正的下一页）
            # 或者：优先保留包含 "page" 或数字的链接
            if chunk_result.get("next_page_url"):
                new_next = chunk_result["next_page_url"]
                # 简单的去重/优先级逻辑：如果之前没找到，或者新找到的看起来更像分页
                if not detected_next_page:
                    detected_next_page = new_next
                    print(f"      🔎 第 {i+1} 块发现了翻页链接: {detected_next_page}")
                elif new_next != detected_next_page:
                    # 如果这块也找到了不一样的链接，可能是底部的"下一页"覆盖了中间的"更多"
                    # 通常底部的优先级更高
                    detected_next_page = new_next
                    print(f"      🔄 第 {i+1} 块更新了翻页链接: {detected_next_page}")

        print(f"📦 分块提取完成，原始总条数: {len(all_items)}")

        # 全局去重
        final_items = self._deduplicate_items(all_items)
        
        return {
            "items": final_items,
            "next_page_url": detected_next_page
        }
    
    def _try_extract_next_page_by_regex(self, text: str) -> Union[str, None]:
        """
        【新增】正则兜底提取：当 LLM 忽略时，暴力从 Markdown 中查找导航链接
        针对: [更多 __](https://...) 或 [下一页](...)
        """
        # 关键词：更多, Next, 下一页, 下页, More, >>, »
        keywords = r"(更多|Next|下一页|下页|More|>>|»)"
        
        # Regex 解释:
        # \[\s* 匹配 [ 和空白
        # ([^\]]*?keywords[^\]]*?) 匹配包含关键词的文本 (Group 1: Link Text)
        # \s*\]           匹配 ] 和空白
        # \((https?://[^)]+)\)     匹配 (URL) (Group 2: URL)
        
        pattern = re.compile(r'\[\s*([^\]]*?' + keywords + r'[^\]]*?)\s*\]\((https?://[^)]+)\)', re.IGNORECASE)
        
        matches = pattern.findall(text)
        if matches:
            # 可能会匹配到多个，比如 [更多电影] [更多新闻]
            # 策略：优先返回第一个匹配到的有效 HTTP 链接
            for link_text, kw, url in matches:
                # 排除明显无关的链接
                if "APP" in link_text or "下载" in link_text:
                    continue
                return url
        return None

    def _process_single_chunk(self, chunk_text: str, target: List[str], source: str) -> Dict[str, Any]:
        """
        处理单个块，返回 {"items": [], "next_page_url": ...}
        """
        prompt = PromptTemplate.from_template(SCRAWL_DATA_SYSTEM_PROMPT)
        
        try:
            # user_query 转字符串，避免由列表引发格式问题
            resp = self.llm.invoke(prompt.format(user_query=str(target), summary=chunk_text, source=source))
            content = resp.content.strip()
        except Exception as e:
            print(f"❌ LLM Chunk Error: {e}")
            return {"items": [], "next_page_url": None}

        # 解析 JSON
        raw_result = self._parse_json_safely(content)
        
        # 格式标准化：确保返回结构是 {"items": [], "next_page_url": None}
        final_structure = {"items": [], "next_page_url": None}

        if isinstance(raw_result, dict):
            # 情况 A: 标准返回
            if "items" in raw_result:
                final_structure["items"] = raw_result["items"] if isinstance(raw_result["items"], list) else []
                final_structure["next_page_url"] = raw_result.get("next_page_url")
            # 情况 B: 旧格式单个对象
            elif "items" not in raw_result and "error" not in raw_result: 
                 final_structure["items"] = [raw_result]

        elif isinstance(raw_result, list):
            # 情况 C: 纯列表
            final_structure["items"] = raw_result
        
        # ============================================================
        # 【关键修复】正则兜底检测翻页链接
        # ============================================================
        if not final_structure.get("next_page_url"):
            fallback_url = self._try_extract_next_page_by_regex(chunk_text)
            if fallback_url:
                print(f"🔎 [Regex Fallback] LLM未识别，但正则提取到翻页链接: {fallback_url}")
                final_structure["next_page_url"] = fallback_url

        return final_structure

    def _split_text_by_lines(self, text: str, max_length: int) -> List[str]:
        """按行切分文本，并安全处理超长行"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line) + 1 
            
            if line_len > max_length:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                while len(line) > max_length:
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                
                current_chunk = [line]
                current_length = len(line) + 1
                continue 

            if current_length + line_len > max_length:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def _parse_json_safely(self, text: str) -> Union[List, Dict]:
        """安全解析 JSON"""
        try:
            return json.loads(text)
        except:
            pass

        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except:
            pass

        try:
            match = re.search(r'\{[\s\S]*\}', text) 
            if match:
                return json.loads(match.group(0))
        except:
            pass

        try:
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                return json.loads(match.group(0))
        except:
            pass

        return {"items": [], "next_page_url": None, "error": "Parse Failed"}

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """结果去重"""
        if not items: return []
        unique_items = []
        seen_urls = set()
        target_keys = ["url", "link", "href", "链接", "详情页链接", "文章链接", "full_url"]

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
                normalized = found_url.rstrip('/')
                if normalized in seen_urls: continue
                seen_urls.add(normalized)
                unique_items.append(item)
            else:
                unique_items.append(item)
        
        if len(items) != len(unique_items):
            print(f"🔍 ExtractorAgent 全局去重: {len(items)} -> {len(unique_items)} 条")
        return unique_items
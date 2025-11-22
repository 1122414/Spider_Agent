import os
import re
import html
import json
from typing import List, Dict, Any, Set, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# 修复导入路径，避免 ImportError
from langchain_core.prompts import PromptTemplate
from agent.prompt_template import SCRAWL_DATA_SYSTEM_PROMPT
from config import *

load_dotenv()

# MODA_OPENAI_API_KEY = os.environ.get("MODA_OPENAI_API_KEY")
# MODA_OPENAI_BASE_URL = os.environ.get("MODA_OPENAI_BASE_URL")
# MODEL_NAME = os.environ.get("MODA_MODEL_NAME", "gpt-4o-mini")

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
            # 翻页链接通常在页面的底部（即最后几个块中）
            # 如果后面的块发现了翻页链接，覆盖之前的
            if chunk_result.get("next_page_url"):
                detected_next_page = chunk_result["next_page_url"]
                print(f"      🔎 第 {i+1} 块发现了翻页链接: {detected_next_page}")

        print(f"📦 分块提取完成，原始总条数: {len(all_items)}")

        # 全局去重
        final_items = self._deduplicate_items(all_items)
        
        return {
            "items": final_items,
            "next_page_url": detected_next_page
        }

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
            # 情况 A: 标准返回 {"items": [...], "next_page_url": "..."}
            if "items" in raw_result:
                final_structure["items"] = raw_result["items"] if isinstance(raw_result["items"], list) else []
                final_structure["next_page_url"] = raw_result.get("next_page_url")
            # 情况 B: LLM 还是返回了旧格式的单个对象 (虽然 Prompt 禁止了)
            elif "items" not in raw_result: 
                 # 尝试把整个 dict 当作一个 item，排除 error 字段的情况
                 if "error" not in raw_result:
                     final_structure["items"] = [raw_result]

        elif isinstance(raw_result, list):
            # 情况 C: LLM 返回了纯列表 (旧格式)
            final_structure["items"] = raw_result
        
        return final_structure

    def _split_text_by_lines(self, text: str, max_length: int) -> List[str]:
        """按行切分文本，并安全处理超长行"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line) + 1 # +1 是考虑换行符
            
            # --- 修复开始：处理单行超长的情况 ---
            if line_len > max_length:
                # 1. 先把手头积攒的 current_chunk 存掉
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 2. 循环切分当前这行超长的文本
                # 比如 line 有 70k，max_length 是 30k，这里会切成 30k, 30k, 10k
                while len(line) > max_length:
                    # 切下前 max_length 个字符作为一个单独的 chunk
                    chunks.append(line[:max_length])
                    # 把剩下的部分赋值回 line，继续处理
                    line = line[max_length:]
                
                # 3. 剩下的部分（也就是 < max_length 的部分）不能丢
                # 把它作为新 chunk 的开头，放入 current_chunk
                current_chunk = [line]
                current_length = len(line) + 1
                continue 
            # --- 修复结束 ---

            # 下面是正常行的处理逻辑（保持不变）
            if current_length + line_len > max_length:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        # 处理最后剩下的 residue
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def _parse_json_safely(self, text: str) -> Union[List, Dict]:
        """安全解析 JSON"""
        # 1. 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 2. 清洗 Markdown 代码块标记
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except:
            pass

        # 3. 正则提取：优先尝试提取对象结构 {...} (新 Prompt 要求返回对象)
        try:
            # dotall 模式，让 . 匹配换行符
            match = re.search(r'\{[\s\S]*\}', text) 
            if match:
                return json.loads(match.group(0))
        except:
            pass

        # 4. 正则提取：兜底尝试提取数组 [...] (防止 LLM 返回旧格式)
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
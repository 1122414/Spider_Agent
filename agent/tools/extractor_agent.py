import json
import re
import time
import traceback
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document 
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import Html2TextTransformer

# 引入核心 DOM 工具 (从 agent.tools.dom_helper)
from agent.tools.dom_helper import dom_analyzer
# 引入提示词
from agent.prompt_template import XPATH_ANALYSIS_PROMPT, SCRAWL_DATA_SYSTEM_PROMPT
from config import *

load_dotenv()

class ExtractorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME, 
            temperature=0, # 必须为 0，保证逻辑精准
            openai_api_key=OPENAI_API_KEY, 
            openai_api_base=OPENAI_BASE_URL
        )
        self.html2text = Html2TextTransformer(ignore_links=False)
    def _sanitize_for_llm(self, text: str, aggressive: bool = False) -> str:
        """
        【安全清洗】在发送给 LLM 前清洗文本。
        aggressive=True 时启用强力模式，用于重试。
        """
        if not text: return ""
        
        # 1. 移除控制字符
        text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 2. 移除 Base64 图片
        text = re.sub(r'data:image\/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+', '[BASE64_REMOVED]', text)
        
        # 3. 移除超长字符串 (如加密 Token, CSS chunks)
        text = re.sub(r'[a-zA-Z0-9+/=]{80,}', '[LONG_TOKEN_REMOVED]', text)
        
        if aggressive:
            # 【强力模式】
            # 1. 移除 URL 参数 (往往包含敏感追踪 ID)
            # 匹配 http... ?a=b... 替换为 [URL_PARAM_REMOVED]
            text = re.sub(r'(https?://[^?\s]+)\?[^\s]*', r'\1?[PARAMS_REMOVED]', text)
            
            # 2. 移除所有脚本/样式遗留 (以防万一)
            text = re.sub(r'<script.*?>.*?</script>', '[SCRIPT_REMOVED]', text, flags=re.DOTALL)
            text = re.sub(r'<style.*?>.*?</style>', '[STYLE_REMOVED]', text, flags=re.DOTALL)
            
            # 3. 过滤掉非 ASCII 且非中日韩字符的怪异符号 (Emoji 除外)
            # 这里简单处理：如果一行里乱码太多，直接丢弃该行? 
            # 暂时只做 URL 清洗，通常这就够了。
            
        return text

    def get_content(self, fetched_html: str, target: List[str], source: str, max_nodes: int = 200) -> Dict[str, Any]:
        """
        数据提取主入口
        """
        # 1. 预检查
        if not fetched_html or len(fetched_html.strip()) < 10:
            print("⚠️ 警告: 输入的 HTML 内容为空或过短，跳过提取。")
            return {"items": [], "next_page_url": None}

        print(f"🏗️ [Extractor] 开始处理 URL: {source}")
        
        # ============================================================
        # 策略 A: 骨架分析法 (优先)
        # ============================================================
        try:
            # 1. 生成骨架
            html_snippet = fetched_html[:80000]
            skeleton = dom_analyzer.summarize_structure(html_snippet, max_nodes=max_nodes)

            # 2. 初次尝试 (标准清洗)
            safe_skeleton = self._sanitize_for_llm(skeleton, aggressive=False)
            
            if len(safe_skeleton) > 100:
                print(f"🦴 DOM 骨架生成完毕 ({len(safe_skeleton)} chars)。请求 LLM 生成 XPath...")
                
                prompt = PromptTemplate.from_template(XPATH_ANALYSIS_PROMPT)
                user_query_str = f"提取字段: {', '.join(target)}"
                
                try:
                    resp = self.llm.invoke(prompt.format(user_query=user_query_str, skeleton=safe_skeleton))
                    json_str = resp.content
                except Exception as e:
                    # 捕捉 400 风控错误
                    error_str = str(e)
                    if "data_inspection_failed" in error_str or "400" in error_str:
                        print("⚠️ 触发内容风控 (Level 1)，尝试强力清洗重试...")
                        # 3. 重试机制 (强力清洗)
                        safe_skeleton_aggressive = self._sanitize_for_llm(skeleton, aggressive=True)
                        try:
                            # 等待 1 秒再重试，避免并发限制
                            time.sleep(1)
                            resp = self.llm.invoke(prompt.format(user_query=user_query_str, skeleton=safe_skeleton_aggressive))
                            json_str = resp.content
                            print("✅ 强力清洗后 LLM 请求成功！")
                        except Exception as e2:
                            print(f"❌ 强力清洗后依然失败: {e2}")
                            raise e2 # 抛出异常，触发回退
                    else:
                        raise e # 其他错误直接抛出

                # 处理 LLM 返回
                clean_json = json_str.strip().replace("```json", "").replace("```", "")
                try:
                    rule = json.loads(clean_json)
                    print(f"🎯 LLM 生成 XPath 规则: {json.dumps(rule, ensure_ascii=False)}")
                    
                    extracted_items = dom_analyzer.extract_by_xpath(fetched_html, rule)
                    
                    if extracted_items and len(extracted_items) > 0:
                        print(f"✅ XPath 提取成功! 共 {len(extracted_items)} 条数据")
                        next_url = self._try_extract_next_page_by_regex(fetched_html)
                        if next_url:
                            print(f"      🔎 [Regex] 补充提取到翻页链接: {next_url}")
                        return {
                            "items": extracted_items,
                            "next_page_url": next_url
                        }
                    else:
                        print("⚠️ XPath 规则执行结果为空，尝试回退...")
                except json.JSONDecodeError:
                    print(f"⚠️ XPath 规则解析失败: {resp.content}")
            else:
                print("⚠️ 骨架生成过短，跳过 XPath 策略。")
                
        except Exception as e:
            print(f"⚠️ XPath 策略最终执行异常 (已回退): {e}")
            # traceback.print_exc() 

        # ============================================================
        # 策略 B: 文本分块提取 (兜底)
        # ============================================================
        print("🔄 回退到纯文本 LLM 分块提取模式...")
        # 这里的输入 html 也要注意，必须使用强力清洗，否则分块也会挂
        safe_html = self._sanitize_for_llm(fetched_html, aggressive=True)
        return self._extract_by_chunking_strategy(safe_html, target, source)

    # ============================================================
    # 辅助方法 & 兜底逻辑
    # ============================================================
    
    def _extract_by_chunking_strategy(self, fetched_html: str, target: List[str], source: str) -> Dict[str, Any]:
        """
        原有的分块提取逻辑 (Slow Path)
        """
        CHUNK_SIZE = 20000 
        if len(fetched_html) <= CHUNK_SIZE:
            # 【安全修复】分块模式也需要清洗
            safe_html = self._sanitize_for_llm(fetched_html)
            return self._process_single_chunk(safe_html, target, source)
        
        docs = [Document(page_content=safe_html, metadata={"source": source})]
        transformed_docs = self.html2text.transform_documents(docs)
        pure_text = transformed_docs[0].page_content if transformed_docs else ""

        print(f"📦 内容过长 ({len(fetched_html)} chars)，启动分块提取...")
        chunks = self._split_text_by_lines(pure_text, CHUNK_SIZE)
        all_items = []
        detected_next_page = None
        
        for i, chunk in enumerate(chunks):
            # 【安全修复】清洗每一个块
            print(f"📦 分块 {i+1}/{len(chunks)}: {len(chunk)} chars")
            safe_chunk = self._sanitize_for_llm(chunk)
            chunk_result = self._process_single_chunk(safe_chunk, target, source)
            items = chunk_result.get("items", [])
            if items: all_items.extend(items)
            if chunk_result.get("next_page_url"): detected_next_page = chunk_result["next_page_url"]

        return {
            "items": self._deduplicate_items(all_items),
            "next_page_url": detected_next_page
        }

    def _process_single_chunk(self, chunk_text: str, target: List[str], source: str) -> Dict[str, Any]:
        """处理单个文本块"""
        prompt = PromptTemplate.from_template(SCRAWL_DATA_SYSTEM_PROMPT)
        try:
            resp = self.llm.invoke(prompt.format(user_query=str(target), summary=chunk_text, source=source))
            content = resp.content.strip()
        except Exception as e:
            return {"items": [], "next_page_url": None}

        raw_result = self._parse_json_safely(content)
        final_structure = {"items": [], "next_page_url": None}

        if isinstance(raw_result, dict):
            if "items" in raw_result:
                final_structure["items"] = raw_result["items"] if isinstance(raw_result["items"], list) else []
                final_structure["next_page_url"] = raw_result.get("next_page_url")
            elif "items" not in raw_result: 
                 final_structure["items"] = [raw_result]
        elif isinstance(raw_result, list):
            final_structure["items"] = raw_result
        
        if not final_structure.get("next_page_url"):
            fallback_url = self._try_extract_next_page_by_regex(chunk_text)
            if fallback_url: final_structure["next_page_url"] = fallback_url

        return final_structure

    def _try_extract_next_page_by_regex(self, text: str) -> Union[str, None]:
        """正则兜底提取翻页链接"""
        keywords = r"(更多|Next|下一页|下页|More|>>|»)"
        pattern = re.compile(r'\[\s*([^\]]*?' + keywords + r'[^\]]*?)\s*\]\((https?://[^)]+)\)', re.IGNORECASE)
        html_pattern = re.compile(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>.*?'+keywords+'.*?</a>', re.IGNORECASE)
        
        matches = pattern.findall(text)
        if matches: return matches[0][2]
        
        html_matches = html_pattern.findall(text)
        if html_matches: return html_matches[0]
        return None

    def _split_text_by_lines(self, text: str, max_length: int) -> List[str]:
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
        if current_chunk: chunks.append("\n".join(current_chunk))
        return chunks

    def _parse_json_safely(self, text: str) -> Union[List, Dict]:
        try: return json.loads(text)
        except: pass
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try: return json.loads(cleaned)
        except: pass
        try:
            match = re.search(r'\{[\s\S]*\}', text) 
            if match: return json.loads(match.group(0))
        except: pass
        try:
            match = re.search(r'\[[\s\S]*\]', text)
            if match: return json.loads(match.group(0))
        except: pass
        return {"items": [], "next_page_url": None}

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        if not items: return []
        unique_items = []
        seen_urls = set()
        target_keys = ["url", "link", "href", "链接"]
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
        return unique_items
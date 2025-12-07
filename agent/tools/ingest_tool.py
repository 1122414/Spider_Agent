import os
import time
import json
import httpx
import uuid
import random
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# LangChain & Milvus
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus

# 引入 registry 以获取缓存数据
from agent.tools.registry import tool_registry

from config import *

load_dotenv()

def get_embedding_model():
    """
    工厂函数：自动选择 OpenAI 或 Ollama 嵌入模型
    """
    http_client = httpx.Client(trust_env=False, timeout=60.0)
    if EMBEDDING_TYPE == 'local_ollama':
        print(f"🔌 使用 OllamaEmbeddings (Model: {OPENAI_OLLAMA_EMBEDDING_MODEL})...")
        # OllamaEmbeddings 不需要 /v1 后缀
        base_url = OPENAI_OLLAMA_BASE_URL.replace("/api/generate", "").replace("/v1", "").rstrip("/")
        return OllamaEmbeddings(
            base_url=base_url,
            model=OPENAI_OLLAMA_EMBEDDING_MODEL
        )
    elif EMBEDDING_TYPE == 'local_vllm':
        print(f"🔌 使用 Vllm OpenAIEmbeddings (Model: {VLLM_OPENAI_EMBEDDING_MODEL})...")
        return OpenAIEmbeddings(
            model=VLLM_OPENAI_EMBEDDING_MODEL,
            openai_api_key=VLLM_OPENAI_EMBEDDING_API_KEY,
            openai_api_base=VLLM_OPENAI_EMBEDDING_BASE_URL,
            http_client=http_client,
            # 关闭本地 Token 检查，强制发送纯文本
            check_embedding_ctx_length=False
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_OLLAMA_BASE_URL
        )

def _resolve_data(data: Union[Dict, List, None]) -> Union[Dict, List, None]:
    cached_data = tool_registry.last_execution_result
    if cached_data:
        if isinstance(cached_data, str):
            print(f"⚠️ [SaveToKB] 缓存数据是字符串，跳过。")
        elif isinstance(cached_data, list) and len(cached_data) == 0:
            print("⚠️ [SaveToKB] 缓存数据为空。")
            return cached_data
        else:
            print(f"🔎 [SaveToKB] 使用内存缓存数据。")
            return cached_data
    return data if data else None

def _extract_items_from_structure(data: Any) -> List[Dict]:
    """
    【核心逻辑】递归查找嵌套字典中包含数据的列表
    解决类似 target_content -> items 这种深层嵌套问题
    """
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        # 1. 优先查找常见的数据容器 Key
        priority_keys = ["items", "data", "list", "target_content", "results", "products"]
        
        for key in priority_keys:
            if key in data:
                val = data[key]
                # 如果找到列表，直接返回
                if isinstance(val, list) and len(val) > 0:
                    return val
                # 如果是字典（如 target_content），递归进去找
                if isinstance(val, dict):
                    deep_items = _extract_items_from_structure(val)
                    if deep_items: 
                        return deep_items
        
        # 2. 如果常见 Key 没找到，就把 Dict 本身当做单条数据
        # 但要排除那种只包含 meta 信息（如 code: 200）的 dict
        if len(data.keys()) > 1: # 至少有点内容的
            return [data]
            
    return []

def _generate_children_summary(children: List[Dict], max_items: int = 50) -> str:
    """
    【新增】生成子项数据的文本摘要
    将 children 列表转换为紧凑的文本块，附加到父文档中。
    """
    if not children:
        return ""
    
    lines = ["\n【关联的详细子项列表 (Children Details)】:"]
    
    # 关键词优化：优先展示 链接、标题 等对用户有用的信息
    priority_keys = ["title", "name", "url", "link", "href", "download", "magnet", "名称", "链接", "下载"]
    
    for i, child in enumerate(children[:max_items]):
        if not isinstance(child, dict): continue
        
        # 提取关键字段
        parts = []
        seen_values = set()  # 【优化】用于去重值
        
        # 1. 先找优先字段
        for pk in priority_keys:
            for k, v in child.items():
                if pk in k.lower() and v and isinstance(v, (str, int)):
                     val_str = str(v).strip()
                     # 防止同一条子项里 "url": "http://..." 和 "link": "http://..." 重复出现
                     if val_str in seen_values: continue
                     if val_str and len(val_str) < 300: # 防止url过长
                        parts.append(f"{k}: {val_str}")
                        seen_values.add(val_str)
        
        # 2. 如果优先字段没找到，稍微补充其他字段（排除 children）
        if not parts:
            for k, v in child.items():
                if k in ["children", "items", "target_content"]: continue
                val_str = str(v).strip()
                if val_str in seen_values: continue
                
                if val_str and len(val_str) < 100:
                    parts.append(f"{k}: {val_str}")
                    seen_values.add(val_str)
                    if len(parts) >= 2: break # 限制长度

        # 格式化单行
        if parts:
            # 去重 parts (虽然上面已有 seen_values，但为了保险)
            unique_parts = list(set(parts))
            lines.append(f"  {i+1}. " + " | ".join(unique_parts))
    
    if len(children) > max_items:
        lines.append(f"  ... (还有 {len(children) - max_items} 条子项数据未展示)")
        
    return "\n".join(lines)

def _flatten_data_to_documents(data: Union[List, Dict], category: str = "general") -> List[Document]:
    """
    通用版数据扁平化：自适应各种字段名
    """
    # 1. 智能提取列表数据
    items = _extract_items_from_structure(data)
        
    if not items:
        print("⚠️ [Flatten] 未找到有效列表数据。")
        return []

    documents = []
    
    for item in items:
        if not isinstance(item, dict): continue
        
        # --- A. 动态识别 Title 和 URL ---
        title = "未命名条目"
        url = ""
        
        # 启发式关键词
        title_keywords = ["title", "name", "名称", "名", "标题", "product", "movie"]
        url_keywords = ["url", "link", "href", "链接", "跳转"]

        # 遍历所有字段来猜测 Title 和 URL
        for k, v in item.items():
            if not isinstance(v, str): continue
            k_lower = k.lower()
            
            # 猜 URL
            if not url and any(kw in k_lower for kw in url_keywords) and (v.startswith("http") or v.startswith("/")):
                url = v
            
            # 猜 Title (优先级：如果 key 包含 keyword，且 value 不像 url)
            if title == "未命名条目" and any(kw in k_lower for kw in title_keywords) and len(v) < 100:
                title = v

        # 如果没猜到 Title，尝试用第一个非 URL 的短字符串作为 Title
        if title == "未命名条目":
            for k, v in item.items():
                if isinstance(v, str) and len(v) > 2 and len(v) < 50 and not v.startswith("http"):
                    title = v
                    break
        
        # 生成一个组 ID，用于关联父子文档（备用）
        group_id = str(uuid.uuid4())

        # 【核心修复】基础 Metadata (包含所有可能字段的默认值，确保 Schema 一致)
        base_metadata = {
            "source": url, 
            "title": title, 
            "category": category,
            "group_id": group_id,
            "has_children": False,  # 默认 False
            "parent_title": ""      # 默认 空字符串
        }

        # --- B. 构建 Parent 文本 (包含 Children 摘要) ---
        content_parts = []
        
        # 【优化】1. 显式添加最重要的标准化信息 (避免重复)
        if title and title != "未命名条目":
            content_parts.append(f"标题: {title}")
        if url:
            content_parts.append(f"链接: {url}")
            
        # 【优化】2. 定义黑名单，过滤掉已知的冗余同义词字段
        # 这些字段通常是 url 或 title 的重复
        redundant_keys = {
            "url", "link", "href", "链接", "跳转链接", "详情页链接", "文章链接", "full_url", "source",
            "title", "name", "名称", "名", "标题", "product_name", "movie_name"
        }
        
        for k, v in item.items():
            if k in ["children", "target_content", "items"] or v is None: 
                continue
                
            k_lower = k.lower()
            
            # 如果 Key 在冗余黑名单里，直接跳过 (因为上面已经添加了标准化的"标题"和"链接")
            if k_lower in redundant_keys:
                continue
                
            val_str = str(v).strip()
            if not val_str: continue
            
            # 二次检查：防止漏网之鱼（Key 不在黑名单，但 Value 与 标题或链接 完全一致）
            if val_str == title or val_str == url:
                continue
                
            content_parts.append(f"{k}: {val_str}")
        
        parent_text_body = "\n".join(content_parts)
        
        # 生成 Children 摘要并附加到 Parent 文本中
        children = item.get("children", [])
        children_summary_text = ""
        if isinstance(children, list) and children:
            children_summary_text = _generate_children_summary(children)
            
        # 组合最终的 Parent 文本
        full_parent_text = parent_text_body
        if children_summary_text:
            full_parent_text += f"\n{children_summary_text}"
        
        if full_parent_text and len(full_parent_text) > 5:
            # 合并 metadata
            meta = base_metadata.copy()
            meta["type"] = "parent_info"
            meta["has_children"] = bool(children)
            # parent_title 默认为空，保持一致
            
            doc = Document(
                page_content=full_parent_text,
                metadata=meta
            )
            documents.append(doc)
            
        # --- C. 递归处理 Children (依旧生成独立的子文档，用于精细检索) ---
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict): continue
                
                child_parts = []
                # 【优化】子项也做同样的去重
                seen_child_values = set()
                
                for k, v in child.items():
                    k_lower = k.lower()
                    if k_lower in redundant_keys: continue
                    
                    val_str = str(v).strip()
                    if val_str and val_str not in seen_child_values:
                        child_parts.append(f"{k}: {val_str}")
                        seen_child_values.add(val_str)
                
                if child_parts:
                    # 子文档带上父标题上下文
                    child_text = f"《{title}》的子项详情:\n" + "\n".join(child_parts)
                    
                    # 合并 metadata
                    child_meta = base_metadata.copy()
                    child_meta["type"] = "child_detail"
                    child_meta["parent_title"] = title
                    # has_children 默认为 False，保持一致
                    
                    child_doc = Document(
                        page_content=child_text,
                        metadata=child_meta
                    )
                    documents.append(child_doc)
                    
    return documents

def save_to_milvus(data: Union[Dict, List] = None, category: str = "general") -> str:
    """
    将数据存入 Milvus 向量知识库 (稳健版)
    """
    actual_data = _resolve_data(data)
    if not actual_data:
        return "保存失败: 没有有效数据"

    # 转换数据
    docs = _flatten_data_to_documents(actual_data, category=category)

    # 过滤空文档
    valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    
    if not valid_docs:
        return f"保存失败: 数据转换后为空 (原始数据类型: {type(actual_data)})"
    
    print(f"🔄 准备处理 {len(valid_docs)} 条数据片段...")

    try:
        embeddings = get_embedding_model()

        # 手动计算向量 (防止 429)
        text_embeddings = []
        metadatas = []
        texts = []
        
        print(f"⚡ 开始计算向量 (Manual Embedding Mode)...")
        for i, doc in enumerate(valid_docs):
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    clean_text = doc.page_content.replace("\n", " ")
                    # 简单的长度截断，防止超出 embedding 模型限制 (例如 OpenAI 是 8191 tokens)
                    if len(clean_text) > 30000: 
                        clean_text = clean_text[:30000]
                        
                    vector = embeddings.embed_query(clean_text)
                    
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)
                    text_embeddings.append(vector)
                    
                    if (i + 1) % 5 == 0:
                        print(f"   -> 已向量化 {i + 1}/{len(valid_docs)} 条")
                    
                    time.sleep(0.05)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str:
                        wait_time = 2 ** retry_count
                        print(f"   ⚠️ 限流等待 {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"   ❌ 第 {i} 条嵌入失败 (Fatal): {e}")
                        break
            
        if not text_embeddings:
            return "保存失败: 所有数据向量化均失败。"

        index_params = {
            "metric_type": "IP",         # 推荐: RAG用 "IP" 或 "COSINE"
            "index_type": "HNSW",        # 索引类型
            "params": {
                "M": 16,                 # 节点最大连接数
                "efConstruction": 250    # 索引构建深度
            }
        }

        print(f"✅ 向量计算完成 ({len(text_embeddings)} 条)，准备存入 Milvus...")

        # 注意：如果你不想每次都清空旧数据，请将 drop_old=False
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
            collection_name=COLLECTION_NAME,
            auto_id=True,
            drop_old=False, 
            index_params=index_params
        )
        
        vector_store.add_embeddings(
            texts=texts,
            embeddings=text_embeddings,
            metadatas=metadatas
        )
        
        try:
            if hasattr(vector_store, 'col') and vector_store.col:
                vector_store.col.flush()
            elif hasattr(vector_store, 'collection') and vector_store.collection:
                vector_store.collection.flush()
        except:
            pass
        
        return f"成功将 {len(text_embeddings)} 条数据存入知识库 (Collection: {COLLECTION_NAME})"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"向量数据库操作失败: {str(e)}"
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Set, Union
import re
from urllib.parse import urljoin

# 引入原生 Playwright
from playwright.async_api import async_playwright
from langchain_core.documents import Document 
from langchain_community.document_transformers import Html2TextTransformer
from agent.tools.extractor_agent import ExtractorAgent

# 应用 nest_asyncio 补丁
nest_asyncio.apply()

# ==========================================
# 1. 辅助工具函数 (Helpers)
# ==========================================

def _normalize_url(url: str) -> str:
    """
    URL 标准化，用于去重比较。
    """
    if not url:
        return ""
    return url.strip().rstrip("/")

async def _auto_scroll(page, max_scrolls: int):
    """
    模拟人工滚动以触发懒加载
    """
    if max_scrolls <= 0:
        return
    print(f"   Start auto-scroll (Max: {max_scrolls})...")
    for i in range(max_scrolls):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500) 
            # print(f"   Scrolled {i+1}/{max_scrolls}")
        except Exception as e:
            print(f"   Scroll failed: {e}")
            break

# ==========================================
# 2. 核心异步逻辑 (Async Core Functions)
# ==========================================

async def playwright_fetch(
    url: str, 
    target: List[str], 
    wait: float = 2.0, 
    max_scrolls: int = 0
) -> Dict:
    """
    [基础爬虫] 使用 Playwright 提取单页面内容
    """
    print(f"🕷️ Fetching: {url}")
    
    raw_html = ""
    error_msg = None

    try:
        async with async_playwright() as p:
            # 生产环境建议 headless=True
            browser = await p.chromium.launch(headless=False) 
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                if max_scrolls > 0:
                    await _auto_scroll(page, max_scrolls)
                else:
                    await page.wait_for_timeout(wait * 1000)

            except Exception as e:
                print(f"⚠️ Page load warning: {e}")
            
            raw_html = await page.content()
            await browser.close()

    except Exception as e:
        error_msg = f"Playwright Critical Error: {str(e)}"
        return {"url": url, "error": error_msg}

    if not raw_html:
        return {"url": url, "error": "Failed to load content"}

    # --- 数据清洗与提取 ---
    docs = [Document(page_content=raw_html, metadata={"source": url})]
    html2text = Html2TextTransformer(ignore_links=False)
    transformed_docs = html2text.transform_documents(docs)
    pure_text = transformed_docs[0].page_content if transformed_docs else ""

    match = re.search(r"<title>(.*?)</title>", raw_html, re.S | re.I)
    title = match.group(1).strip() if match else "No Title"

    try:
        extractor = ExtractorAgent()
        target_content = extractor.get_content(pure_text, target, url)
    except Exception as e:
        target_content = f"Extraction Failed: {str(e)}"

    return {
        "url": url,
        "title": title,
        "target_content": target_content
    }

async def _recursive_crawl_logic(
    current_url: str,
    pipelines: List[List[str]], # 每一层的提取目标
    current_depth: int,
    max_items: int,
    visited_urls: Set[str]
) -> Union[List[Dict], Dict, str]:
    """
    [内部递归函数] 处理多层级爬取逻辑
    """
    # 1. 边界检查
    if current_depth >= len(pipelines):
        return None # 超过预设深度，停止

    target = pipelines[current_depth]
    # 只有列表页才需要滚动，详情页通常不需要
    scrolls = 1 if current_depth == 0 else 0 
    
    # 2. 爬取当前层
    # 自动给每一层加上 "link" 相关的提取提示，方便下一层钻取
    enhanced_target = target + ["link", "url", "href", "链接", "跳转链接"]
    
    fetch_result = await playwright_fetch(current_url, enhanced_target, max_scrolls=scrolls)
    
    if "error" in fetch_result and fetch_result["error"]:
        return {"error": fetch_result["error"], "url": current_url}

    extracted_data = fetch_result.get("target_content")

    # 3. 如果是最后一层，直接返回数据
    if current_depth == len(pipelines) - 1:
        return extracted_data

    # 4. 准备进入下一层
    # 兼容处理：如果提取结果是单个字典，转为列表统一处理
    items = []
    if isinstance(extracted_data, list):
        items = extracted_data
    elif isinstance(extracted_data, dict):
        items = [extracted_data]
    else:
        # 如果提取结果是纯文本或其他，无法继续深入，直接返回
        return extracted_data

    # 5. 遍历当前层条目，寻找链接进入下一层
    results = []
    count = 0
    
    # 常见的链接字段名
    link_keys = ["link", "url", "href", "链接", "详情页链接", "线路链接", "播放链接", "full_url"]

    for item in items:
        # 超过最大数量限制则停止本层遍历
        if count >= max_items:
            break
            
        processed_item = item.copy() if isinstance(item, dict) else {"raw": item}
        
        # A. 寻找下一层链接
        next_url = None
        if isinstance(item, dict):
            for key in link_keys:
                if key in item and item[key] and isinstance(item[key], str):
                    candidate = item[key].strip()
                    if len(candidate) > 1:
                        next_url = candidate
                        break
        
        # B. 如果找到链接，递归钻取
        if next_url:
            # 拼接完整 URL
            full_next_url = urljoin(current_url, next_url)
            normalized_next = _normalize_url(full_next_url)

            if normalized_next not in visited_urls:
                print(f"   👉 [Depth {current_depth}->{current_depth+1}] Digging: {full_next_url}")
                visited_urls.add(normalized_next)
                
                # 【递归调用】
                sub_data = await _recursive_crawl_logic(
                    full_next_url, 
                    pipelines, 
                    current_depth + 1, 
                    max_items, 
                    visited_urls
                )
                
                # 将下一层数据挂载到当前 item 的 "children" 字段
                # 或者如果下一层返回的是字典（合并），视情况而定。这里统一挂在 children 下结构最清晰。
                processed_item["children"] = sub_data
                count += 1
            else:
                processed_item["info"] = "URL visited or repeated"
        
        results.append(processed_item)

    return results

async def hierarchical_crawl(
    url: str, 
    crawl_scopes: List[List[str]], 
    max_items: int = 3
) -> Dict:
    """
    [多层级深度爬虫 - 异步入口]
    参数:
      url: 起始 URL
      crawl_scopes: 每一层的提取目标列表。
         例如: [ ["动漫标题", "链接"], ["线路链接", "线路名"], ["评论", "视频标题"] ]
      max_items: 每一层最大抓取数量（防止指数级爆炸）
    """
    print(f"🚀 [Multi-Level] 启动多层爬取: {url}")
    print(f"   Pipeline Depth: {len(crawl_scopes)} 层")

    visited_urls = set()
    visited_urls.add(_normalize_url(url))

    # 开始递归
    final_data = await _recursive_crawl_logic(
        url, 
        crawl_scopes, 
        current_depth=0, 
        max_items=max_items, 
        visited_urls=visited_urls
    )

    return {
        "root_url": url,
        "depth_configured": len(crawl_scopes),
        "data": final_data
    }

# ==========================================
# 3. 同步包装器 (Sync Wrappers)
# ==========================================

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)

def sync_playwright_fetch(url: str, target: List[str], max_scrolls: int = 0) -> Dict:
    """基础爬虫入口"""
    return _run_async(playwright_fetch(url, target, max_scrolls=max_scrolls))

def sync_hierarchical_crawl(url: str, crawl_scopes: List[List[str]], max_items: int = 3) -> Dict:
    """
    多层级爬虫入口，支持任意层级
    """
    return _run_async(hierarchical_crawl(url, crawl_scopes, max_items))
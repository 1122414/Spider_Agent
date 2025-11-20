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
    返回结构包含 extractor 的原始返回 (items + next_page_url)
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
        # ExtractorAgent 现在返回 {"items": [...], "next_page_url": ...}
        target_content = extractor.get_content(pure_text, target, url)
    except Exception as e:
        target_content = {"items": [], "next_page_url": None, "error": str(e)}

    return {
        "url": url,
        "title": title,
        "target_content": target_content
    }

async def _recursive_crawl_logic(
    start_url: str,
    pipelines: List[List[str]], # 每一层的提取目标
    current_depth: int,
    max_items: int,
    visited_urls: Set[str],
    max_pages: int = 3  # 新增：最大翻页数
) -> Union[List[Dict], Dict, str]:
    """
    [内部递归函数] 处理多层级爬取逻辑，支持翻页
    """
    # 1. 边界检查
    if current_depth >= len(pipelines):
        return None 

    target = pipelines[current_depth]
    # 只有列表页(Depth 0)或明确需要翻页的层级才滚动
    scrolls = 1 
    
    # 自动给每一层加上链接提取提示
    enhanced_target = target + ["link", "url", "href", "链接", "跳转链接"]
    
    all_layer_results = []
    current_page_url = start_url
    page_count = 0

    # ============================
    # 分页循环 (Pagination Loop)
    # ============================
    while current_page_url and page_count < max_pages:
        # 去重检查 (针对列表页本身)
        normalized_current = _normalize_url(current_page_url)
        if normalized_current in visited_urls:
             print(f"   ⚠️ [Depth {current_depth}] Page visited, stopping pagination: {current_page_url}")
             break
        visited_urls.add(normalized_current)

        if page_count > 0:
            print(f"   📄 [Depth {current_depth}] Flipping to Page {page_count + 1}: {current_page_url}")

        # 2. 爬取当前页
        fetch_result = await playwright_fetch(current_page_url, enhanced_target, max_scrolls=scrolls)
        
        if "error" in fetch_result and fetch_result["error"]:
            print(f"   ❌ Fetch error at {current_page_url}: {fetch_result['error']}")
            break

        extracted_data = fetch_result.get("target_content", {})
        
        # 兼容性处理：确保拿到 items 列表和 next_page_url
        items = []
        next_link = None

        if isinstance(extracted_data, dict):
            items = extracted_data.get("items", [])
            next_link = extracted_data.get("next_page_url")
            # 如果旧版 extractor 返回了 content 放在其他字段，做个兼容（视 Extractor 实现而定）
        elif isinstance(extracted_data, list):
            items = extracted_data # 旧版兼容
        
        # 3. 处理当前页的 items
        # 如果不是最后一层，需要递归深入
        if current_depth < len(pipelines) - 1:
            processed_items = await _process_items_recursively(
                items, 
                current_page_url, 
                pipelines, 
                current_depth, 
                max_items, 
                visited_urls,
                max_pages
            )
            all_layer_results.extend(processed_items)
        else:
            # 最后一层，直接收集数据
            all_layer_results.extend(items)

        # 4. 准备下一页
        if next_link:
            # 拼接完整 URL
            next_full_url = urljoin(current_page_url, next_link)
            
            # 防止原地踏步
            if _normalize_url(next_full_url) == normalized_current:
                print("   ⚠️ Next page is same as current, stopping.")
                break
                
            current_page_url = next_full_url
            page_count += 1
        else:
            # 没有下一页了
            break
    
    return all_layer_results

async def _process_items_recursively(
    items: List[Dict], 
    base_url: str,
    pipelines: List[List[str]],
    current_depth: int,
    max_items: int,
    visited_urls: Set[str],
    max_pages: int
) -> List[Dict]:
    """
    辅助函数：遍历 items 并递归调用下一层
    """
    results = []
    count = 0
    link_keys = ["link", "url", "href", "链接", "详情页链接", "线路链接", "播放链接", "full_url"]

    for item in items:
        if not isinstance(item, dict):
            results.append({"raw": item})
            continue
            
        if count >= max_items:
            break
            
        processed_item = item.copy()
        
        # A. 寻找下一层链接
        next_url = None
        for key in link_keys:
            if key in item and item[key] and isinstance(item[key], str):
                candidate = item[key].strip()
                if len(candidate) > 1:
                    next_url = candidate
                    break
        
        # B. 递归钻取
        if next_url:
            full_next_url = urljoin(base_url, next_url)
            normalized_next = _normalize_url(full_next_url)

            if normalized_next not in visited_urls:
                print(f"   👉 [Depth {current_depth}->{current_depth+1}] Digging: {full_next_url}")
                # 注意：这里不需要把详情页加入 visited_urls 也是可以的，取决于是否允许不同列表项指向同一详情页
                # 这里加入是为了防环
                visited_urls.add(normalized_next)
                
                sub_data = await _recursive_crawl_logic(
                    full_next_url, 
                    pipelines, 
                    current_depth + 1, 
                    max_items, 
                    visited_urls,
                    max_pages
                )
                
                processed_item["children"] = sub_data
                count += 1
            else:
                processed_item["info"] = "URL visited or repeated"
        
        results.append(processed_item)
    
    return results

async def hierarchical_crawl(
    url: str, 
    crawl_scopes: List[List[str]], 
    max_items: int = 3,
    max_pages: int = 3
) -> Dict:
    """
    [多层级深度爬虫 - 异步入口]
    参数:
      url: 起始 URL
      crawl_scopes: 提取目标二维数组
      max_items: 每一层递归抓取的最大条目数
      max_pages: 每一层列表页的最大翻页数
    """
    print(f"🚀 [Multi-Level] 启动多层爬取: {url}")
    print(f"   Pipeline Depth: {len(crawl_scopes)} 层 | Max Pages: {max_pages}")

    visited_urls = set()
    # visited_urls.add(_normalize_url(url)) # 移到递归内部处理，防止第一页就被跳过

    # 开始递归
    final_data = await _recursive_crawl_logic(
        url, 
        pipelines=crawl_scopes, 
        current_depth=0, 
        max_items=max_items, 
        visited_urls=visited_urls,
        max_pages=max_pages
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

def sync_hierarchical_crawl(url: str, crawl_scopes: List[List[str]], max_items: int = 3, max_pages: int = 3) -> Dict:
    """
    [新版] 多层级爬虫入口，支持翻页参数
    """
    return _run_async(hierarchical_crawl(url, crawl_scopes, max_items, max_pages))
import re
import random
import asyncio
import nest_asyncio
import os
from typing import List, Dict, Any, Set, Union, Optional
from urllib.parse import urljoin

# 引入原生 Playwright
from playwright.async_api import async_playwright, BrowserContext, Page
from langchain_core.documents import Document 
from langchain_community.document_transformers import Html2TextTransformer

# 自定义 Agent
from agent.tools.extractor_agent import ExtractorAgent

# 应用 nest_asyncio 补丁 (防止 Jupyter/EventLoop 冲突)
nest_asyncio.apply()

# ==========================================
# 1. 持久化爬虫类 (Persistent Fetcher)
# ==========================================

class PersistentFetcher:
    def __init__(self, user_data_dir: str = "./browser_data", headless: bool = False):
        """
        初始化持久化爬虫
        :param user_data_dir: 浏览器数据存储路径 (Cookies/Cache将保存在此)
        :param headless: 是否无头模式
        """
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        
        # 初始化工具 (避免每次 fetch 都重新创建)
        self.html2text = Html2TextTransformer(ignore_links=False)
        # 如果 ExtractorAgent 有状态或初始化开销大，建议放在这里
        self.extractor = ExtractorAgent() 

    async def start(self):
        """启动浏览器并加载持久化上下文"""
        if not self.playwright:
            print(f"🚀 Starting persistent browser in: {self.user_data_dir}")
            self.playwright = await async_playwright().start()
            
            # 使用 launch_persistent_context 自动保存状态
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                headless=self.headless,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                args=["--disable-blink-features=AutomationControlled"]
            )

    async def stop(self):
        """关闭浏览器资源"""
        if self.context:
            print("🛑 Closing browser context...")
            await self.context.close()
            self.context = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    async def _auto_scroll(self, page: Page, max_scrolls: int):
        """模拟人工滚动以触发懒加载"""
        if max_scrolls <= 0:
            return
        print(f"   Start auto-scroll (Max: {max_scrolls})...")
        for i in range(max_scrolls):
            try:
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                # 随机等待，模拟人类行为
                await page.wait_for_timeout(random.randint(2000, 5000))
            except Exception as e:
                print(f"   Scroll failed: {e}")
                break

    async def fetch(self, url: str, target: List[str], wait: float = 2.0, max_scrolls: int = 0, wait_time: int = 1000, max_nodes: int = 200) -> Dict:
        """
        执行单页面抓取 (复用已打开的浏览器)
        """
        if not self.context:
            await self.start()

        print(f"🕷️ Fetching: {url}")
        
        # 创建新标签页而不是新浏览器
        page = await self.context.new_page() 
        
        raw_html = ""
        error_msg = None
        target_content = {}

        try:
            try:
                # 设置页面加载超时
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)

                await page.wait_for_timeout(random.randint(wait_time, wait_time + 3000))
                
                if max_scrolls > 0:
                    await self._auto_scroll(page, max_scrolls)
                else:
                    await page.wait_for_timeout(wait * 1000)

                raw_html = await page.content()

            except Exception as e:
                print(f"⚠️ Page load warning: {e}")
                # 即使报错，尝试获取已加载的内容
                raw_html = await page.content()

        except Exception as e:
            error_msg = f"Playwright Critical Error: {str(e)}"
        
        finally:
            # 关键：只关闭 Page，不关闭 Context
            await page.close()
            pass

        if error_msg:
            return {"url": url, "error": error_msg}

        if not raw_html:
            return {"url": url, "error": "Failed to load content"}

        # --- 数据清洗与提取 ---
        docs = [Document(page_content=raw_html, metadata={"source": url})]

        # extrator_agent_bck 原版使用
        # transformed_docs = self.html2text.transform_documents(docs)
        # pure_text = transformed_docs[0].page_content if transformed_docs else ""

        # extrator_agent 新版
        match = re.search(r"<title>(.*?)</title>", raw_html, re.S | re.I)
        title = match.group(1).strip() if match else "No Title"

        try:
            # 使用类成员 extractor
            # ExtractorAgent 返回 {"items": [...], "next_page_url": ...}
            # 旧版
            # target_content = self.extractor.get_content(pure_text, target, url)
            # 新版（测试中，12.8）
            target_content = self.extractor.get_content(raw_html, target, url, max_nodes=max_nodes)
        except Exception as e:
            target_content = {"items": [], "next_page_url": None, "error": str(e)}

        return {
            "url": url,
            "title": title,
            "target_content": target_content
        }

# ==========================================
# 2. 辅助函数 (Helpers)
# ==========================================

def _normalize_url(url: str) -> str:
    if not url:
        return ""
    return url.strip().rstrip("/")

# ==========================================
# 3. 核心递归逻辑 (Refactored for PersistentFetcher)
# ==========================================

async def _recursive_crawl_logic(
    start_url: str,
    pipelines: List[List[str]],
    current_depth: int,
    max_items: int,
    visited_urls: Set[str],
    fetcher: PersistentFetcher,  # 接收 fetcher 实例
    max_pages: int = 3,
    max_scrolls: int = 1,
    wait_time : int = 1000,
    max_nodes: int = 200
) -> Union[List[Dict], Dict, str]:
    """
    [内部递归函数] 处理多层级爬取逻辑，支持翻页
    """
    # 1. 边界检查
    if current_depth >= len(pipelines):
        return None 

    target = pipelines[current_depth]
    
    # 自动给每一层加上链接提取提示
    enhanced_target = target + ["link", "url", "href", "链接", "跳转链接"]
    
    all_layer_results = []
    current_page_url = start_url
    page_count = 0

    # ============================
    # 分页循环 (Pagination Loop)
    # ============================
    while current_page_url and page_count < max_pages:
        # 去重检查
        normalized_current = _normalize_url(current_page_url)
        if normalized_current in visited_urls:
             print(f"   ⚠️ [Depth {current_depth}] Page visited, stopping pagination: {current_page_url}")
             break
        visited_urls.add(normalized_current)

        if page_count > 0:
            print(f"   📄 [Depth {current_depth}] Flipping to Page {page_count + 1}: {current_page_url}")

        # 2. 爬取当前页 (调用 fetcher 实例方法)
        fetch_result = await fetcher.fetch(current_page_url, enhanced_target, max_scrolls=max_scrolls, wait_time=wait_time, max_nodes=max_nodes)

        if "error" in fetch_result and fetch_result["error"]:
            print(f"   ❌ Fetch error at {current_page_url}: {fetch_result['error']}")
            break

        extracted_data = fetch_result.get("target_content", {})
        
        items = []
        next_link = None

        if isinstance(extracted_data, dict):
            items = extracted_data.get("items", [])
            next_link = extracted_data.get("next_page_url")
        elif isinstance(extracted_data, list):
            items = extracted_data # 旧版兼容
        
        # 3. 处理当前页的 items (递归入口)
        if current_depth < len(pipelines) - 1:
            processed_items = await _process_items_recursively(
                items, 
                current_page_url, 
                pipelines, 
                current_depth, 
                max_items, 
                visited_urls,
                fetcher, # 传递 fetcher
                max_pages
            )
            all_layer_results.extend(processed_items)
        else:
            # 最后一层，直接收集数据
            all_layer_results.extend(items)

        # 4. 准备下一页
        if not next_link or not isinstance(next_link, str):
            print(f"[Warning] 跳过无效链接: {next_link}")
            continue  # 或者 return，取决于你的循环结构

        # 2. 检查 current_page_url (虽然可能性较小，但也可能是 None)
        if not current_page_url or not isinstance(current_page_url, str):
            print(f"[Error] 当前页面 URL 无效: {current_page_url}")
            continue
        
        if next_link:
            next_full_url = urljoin(current_page_url, next_link)
            if _normalize_url(next_full_url) == normalized_current:
                print("   ⚠️ Next page is same as current, stopping.")
                break
            current_page_url = next_full_url
            page_count += 1
        else:
            break
    
    return all_layer_results

async def _process_items_recursively(
    items: List[Dict], 
    base_url: str,
    pipelines: List[List[str]],
    current_depth: int,
    max_items: int,
    visited_urls: Set[str],
    fetcher: PersistentFetcher, # 接收 fetcher
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
                
                sub_data = await _recursive_crawl_logic(
                    full_next_url, 
                    pipelines, 
                    current_depth + 1, 
                    max_items, 
                    visited_urls,
                    fetcher, # 传递 fetcher
                    max_pages
                )

                # 防止死循环，将详情页也加入 visited
                visited_urls.add(normalized_next)
                
                processed_item["children"] = sub_data
                count += 1
            else:
                processed_item["info"] = "URL visited or repeated"
        
        results.append(processed_item)
    
    return results

# ==========================================
# 4. 异步入口 (Entry Point)
# ==========================================

async def hierarchical_crawl(
    url: str, 
    crawl_scopes: List[List[str]], 
    max_items: int = 3,
    max_pages: int = 3,
    max_scrolls: int = 1,
    headless: bool = False, # 暴露 headless 参数
    wait_time: int = 1000,
    max_nodes: int = 200
) -> Dict:
    """
    [多层级深度爬虫 - 异步入口]
    """
    print(f"🚀 [Multi-Level] 启动多层爬取: {url}")
    print(f"   Pipeline Depth: {len(crawl_scopes)} 层 | Max Pages: {max_pages}")

    # 1. 初始化 Fetcher
    fetcher = PersistentFetcher(headless=headless)
    
    visited_urls = set()
    final_data = []

    try:
        # 2. 启动浏览器 (整个任务只启动这一次)
        await fetcher.start()

        # 3. 开始递归逻辑
        final_data = await _recursive_crawl_logic(
            url, 
            pipelines=crawl_scopes, 
            current_depth=0, 
            max_items=max_items, 
            visited_urls=visited_urls,
            fetcher=fetcher, # 注入 fetcher
            max_pages=max_pages,
            max_scrolls=max_scrolls,
            wait_time=wait_time,
            max_nodes=max_nodes
        )
    except Exception as e:
        print(f"❌ Critical Error during crawl: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 任务结束，关闭浏览器
        await fetcher.stop()

    return {
        "root_url": url,
        "depth_configured": len(crawl_scopes),
        "data": final_data
    }

# ==========================================
# 5. 同步包装器 (Sync Wrappers)
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
    
# 添加到 agent/tools/crawl_tool.py 末尾

def sync_playwright_fetch(url: str, target: List[str], max_scrolls: int = 0, headless: bool = False, wait_time: int = 1000, max_nodes: int = 200) -> Dict:
    """
    [同步包装器] 基础单页面抓取 (复用 PersistentFetcher)
    """
    async def _runner():
        # 为了单次调用也享受持久化，我们这里临时实例化一个 fetcher
        fetcher = PersistentFetcher(headless=headless)
        try:
            await fetcher.start()
            return await fetcher.fetch(url, target, max_scrolls=max_scrolls, wait_time=wait_time, max_nodes=max_nodes)
        finally:
            await fetcher.stop()
            
    return _run_async(_runner())

def sync_hierarchical_crawl(
    url: str, 
    crawl_scopes: List[List[str]], 
    max_items: int = 3, 
    max_pages: int = 3, 
    max_scrolls: int = 1,
    headless: bool = False,
    wait_time: int = 1000,
    max_nodes: int = 200
) -> Dict:
    """
    [新版] 多层级爬虫同步入口
    """
    return _run_async(hierarchical_crawl(url, crawl_scopes, max_items, max_pages, max_scrolls, headless, wait_time, max_nodes))

# 使用示例 (可选)
if __name__ == "__main__":
    # 示例配置
    start_url = "https://example.com/list"
    scopes = [
        ["电影名称", "评分"],         # 第一层: 列表页
        ["剧情简介", "下载地址"]      # 第二层: 详情页
    ]
    
    # 运行
    # result = sync_hierarchical_crawl(start_url, scopes, max_items=2, headless=False)
    # print(result)
    pass
import asyncio
import nest_asyncio
from typing import List, Dict
import re

# 引入原生 Playwright
from playwright.async_api import async_playwright
# 引入 Document 对象以兼容后续的 Html2TextTransformer
from langchain_core.documents import Document 
from langchain_community.document_transformers import Html2TextTransformer
from agent.tools.extractor_agent import ExtractorAgent

# 应用 nest_asyncio 补丁
nest_asyncio.apply()

async def playwright_fetch(url: str, target: List[str], wait: float = 2.0) -> Dict:
    """
    使用原生 Playwright 提取内容，绕过 LangChain Loader 的参数限制
    """
    print(f"Running playwright_fetch for: {url}")
    
    raw_html = ""
    error_msg = None

    # --- 1. 原生 Playwright 执行逻辑 ---
    try:
        async with async_playwright() as p:
            # 这里可以自由设置 headless=False 来显示浏览器
            browser = await p.chromium.launch(headless=False) 
            
            # 创建上下文，设置 UserAgent 防止被拦截
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()

            # 【核心修复】：直接调用 page.goto 并传入自定义参数
            # wait_until='domcontentloaded' 只要HTML加载完就继续，不傻等图片
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                # 如果超时但页面已加载了部分内容，我们尝试继续
                print(f"Warning: Page load timeout or error: {e}")
            
            # 可选：稍微等待一下动态内容渲染（如果需要）
            # await page.wait_for_timeout(2000) 

            # 获取网页 HTML
            raw_html = await page.content()
            await browser.close()

    except Exception as e:
        error_msg = f"Playwright Critical Error: {str(e)}"
        return {"url": url, "error": error_msg}

    if not raw_html:
        return {"url": url, "error": "Failed to load content (Empty HTML)"}

    # --- 2. 封装为 Document 对象以适配 LangChain 工具链 ---
    # Html2TextTransformer 需要输入 Document 对象列表
    docs = [Document(page_content=raw_html, metadata={"source": url})]

    # --- 3. 转纯文本 ---
    html2text = Html2TextTransformer(ignore_links=False)
    transformed_docs = html2text.transform_documents(docs)
    pure_text = transformed_docs[0].page_content if transformed_docs else ""

    # --- 4. 提取标题 (简单的正则提取) ---
    match = re.search(r"<title>(.*?)</title>", raw_html, re.S | re.I)
    title = match.group(1).strip() if match else "No Title"

    # --- 5. 提取用户目标字段 ---
    try:
        extractor = ExtractorAgent()
        target_content = extractor.get_content(pure_text, target, url)
    except Exception as e:
        target_content = f"Extraction Failed: {str(e)}"

    return {
        "url": url,
        # "html": raw_html[:50000], # 如果不想返回巨大HTML，可以取消注释这行截断
        "html": raw_html,
        "title": title,
        "pure_text": pure_text,
        "target_content": target_content
    }

def sync_playwright_fetch(url: str, target: List[str]) -> Dict:
    """
    同步包装器：自动处理 Event Loop 冲突
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.run_until_complete(playwright_fetch(url, target))
    else:
        return asyncio.run(playwright_fetch(url, target))
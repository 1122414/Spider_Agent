# DrissionPage 爬虫工具：打开页面，滚动，提取主文本段落/段落列表
from DrissionPage import WebPage
from bs4 import BeautifulSoup
import time
from typing import Optional, List, Dict

def drission_fetch(url: str, wait: float = 2.0, max_scroll: int = 3) -> Dict:
    """
    使用 DrissionPage 打开页面并返回 HTML 文本和部分元信息
    返回：
      {
        "url": url,
        "html": "...",
        "title": "...",
        "paragraphs": ["p1","p2", ...]
      }
    """
    page = WebPage(chromium_options={"headless": True})
    page.get(url)
    time.sleep(wait)

    # 简单滚动加载
    for _ in range(max_scroll):
        try:
            page.scroll.down()
            time.sleep(0.5)
        except Exception:
            break

    html = page.html
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    # fallback: consider also divs with many text
    if not paragraphs:
        divs = [d.get_text(strip=True) for d in soup.find_all("div") if len(d.get_text(strip=True)) > 80]
        paragraphs = divs[:50]

    page.quit()
    return {"url": url, "html": html, "title": title, "paragraphs": paragraphs}

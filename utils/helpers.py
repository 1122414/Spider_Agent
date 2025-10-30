import requests
from urllib.parse import urlparse

def is_valid_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http","https") and p.netloc != ""
    except:
        return False

def simple_site_search(platform: str, keywords: str):
    # 占位：实现使用 site: 搜索 fetch fallback（可用Bing/Google API）
    return None

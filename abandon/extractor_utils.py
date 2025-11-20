import re
import html
from typing import List, Dict
from bs4 import BeautifulSoup, Tag, NavigableString

# utils/extractor_utils.py
def text_preview(s: str, max_len=60):
    s = re.sub(r'\s+', ' ', s).strip()
    s = html.unescape(s)
    return s[:max_len] + ("…" if len(s) > max_len else "")

def get_attr_str(tag: Tag, attr_name: str):
    val = tag.get(attr_name)
    if not val:
        return ""
    if isinstance(val, list):
        return " ".join(val)
    return str(val)

def compute_bs4_xpath(tag: Tag) -> str:
    """
    Compute an absolute XPath for a BeautifulSoup Tag.
    This uses parent chain and counts sibling indexes for same-name tags.
    """
    path_parts = []
    cur = tag
    # stop at document root
    while cur is not None and isinstance(cur, Tag):
        name = cur.name
        # count index among siblings with same name
        idx = 1
        sibling = cur.previous_sibling
        while sibling is not None:
            if isinstance(sibling, Tag) and sibling.name == name:
                idx += 1
            sibling = sibling.previous_sibling
        path_parts.append(f"{name}[{idx}]")
        cur = cur.parent
    # path_parts currently like ['p[3]', 'div[2]', 'body[1]', 'html[1]']
    path = "/" + "/".join(reversed(path_parts))
    return path

def summarize_structure(html_text: str, max_nodes: int = 400, min_text_len: int = 5) -> List[Dict]:
    """
    Returns a list of node summaries:
    [
      {
        "xpath": "/html/body/div[2]/p[3]",
        "tag": "p",
        "id": "xxx",
        "class": "a b",
        "text_preview": "..."
      }, ...
    ]
    Only keep nodes with some textual content (min_text_len) and limit total nodes.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    nodes = []
    # choose tags likely to contain content
    candidate_tags = ['h1','h2','h3','h4','h5','p','li','div','span','a','td','th']
    count = 0
    for tag in soup.find_all(candidate_tags):
        if not isinstance(tag, Tag):
            continue
        txt = tag.get_text(" ", strip=True)
        if not txt or len(txt) < min_text_len:
            continue
        xpath = compute_bs4_xpath(tag)
        nid = tag.get("id","")
        cls = get_attr_str(tag, "class")
        nodes.append({
            # "xpath": xpath,
            "tag": tag.name,
            "id": nid,
            "class": cls,
            "text_preview": text_preview(txt, max_len=120)
        })
        count += 1
        if count >= max_nodes:
            break
    return nodes

def nodes_to_text_summary(nodes: List[Dict], max_lines: int = 200) -> str:
    """
    Convert node dicts into a concise text summary suitable for LLM.
    Each line: index | tag | id | class | xpath | text_preview
    """
    lines = []
    for i, n in enumerate(nodes[:max_lines]):
        id_part = f"id=\"{n['id']}\"" if n['id'] else ""
        class_part = f"class=\"{n['class']}\"" if n['class'] else ""
        line = f"{i:03d} | <{n['tag']} {id_part} {class_part} | {n['text_preview']}"
        lines.append(line)
    return "\n".join(lines)

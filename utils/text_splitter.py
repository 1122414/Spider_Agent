from typing import List
import re

def sentence_split(text: str) -> List[str]:
    # 简单中文/英文句子切分
    # 首先替换多空格
    text = re.sub(r'\s+', ' ', text).strip()
    # split by punctuation
    parts = re.split(r'(?<=[。！？\?\.\!；;])\s*', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def semantic_chunk_texts(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    基于句子边界进行切分，然后合并句子成 chunk，保证 chunk_size（字符）为近似
    使用滑动重叠 overlap 字符来提升召回
    """
    sents = sentence_split(text)
    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= chunk_size:
            cur += (s if cur=="" else " " + s)
        else:
            if cur:
                chunks.append(cur)
            # if one sentence > chunk_size, split it by chars
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size - overlap):
                    part = s[i:i+chunk_size]
                    chunks.append(part)
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)

    # add overlap by merging adjacent chunks to create sliding windows
    merged = []
    for i, c in enumerate(chunks):
        merged.append(c)
        if i > 0:
            # overlap with previous
            prev = chunks[i-1]
            combined = prev[-overlap:] + " " + c[:overlap] if overlap < len(prev) and overlap < len(c) else prev + " " + c
            merged.append(combined)
    # deduplicate near-equals
    final = []
    seen=set()
    for m in merged:
        key = m[:min(64,len(m))]
        if key not in seen:
            final.append(m)
            seen.add(key)
    return final

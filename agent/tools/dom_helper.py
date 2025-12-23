import re
from typing import List, Dict, Union, Optional
from lxml import html, etree

class DOMAnalyzer:
    def __init__(self):
        # 1. 物理删除的噪音标签
        self.NOISE_TAGS = {
            'script', 'style', 'noscript', 'svg', 'iframe', 'meta', 'link', 'head', 
            'path', 'br', 'hr', 'input', 'button', 'form' 
        }
        # 2. 逻辑跳过的容器
        self.SKIP_CONTAINERS_TAGS = {'nav', 'header', 'footer'}
        # 3. 疑似轮播图/干扰项的 Class 关键词
        self.CAROUSEL_KEYWORDS = {'swiper', 'slider', 'carousel', 'banner', 'recommend', 'ad-'}

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text: return ""
        return re.sub(r'\s+', ' ', text).strip()

    def _prune_tree(self, tree: html.HtmlElement) -> html.HtmlElement:
        """DOM 树物理剪枝"""
        etree.strip_elements(tree, *self.NOISE_TAGS, with_tail=False)
        for comment in tree.xpath('//comment()'):
            comment.getparent().remove(comment)
        return tree

    def summarize_structure(self, html_content: str, max_nodes: int = 1000, max_text_len: int = 25) -> str:
        """生成网页 DOM 骨架摘要"""
        if not html_content: return ""
        try:
            tree = html.fromstring(html_content)
        except Exception as e:
            return ""

        tree = self._prune_tree(tree)
        skeleton_lines = []
        node_count = 0
        
        for element in tree.iter():
            if node_count >= max_nodes:
                skeleton_lines.append(f"... (剩余节点已省略，共截取前 {max_nodes} 个节点)")
                break

            if not isinstance(element, html.HtmlElement): continue

            tag = element.tag
            
            # --- 智能跳过逻辑 ---
            if tag in self.SKIP_CONTAINERS_TAGS:
                # 仅保留占位符，不展开
                node_count += 1
                continue
            
            # 检查父级是否在黑名单
            parent = element.getparent()
            in_skip_zone = False
            while parent is not None:
                if parent.tag in self.SKIP_CONTAINERS_TAGS:
                    in_skip_zone = True
                    break
                parent = parent.getparent()
            if in_skip_zone: continue

            # --- 提取核心信息 ---
            elem_id = element.get('id', '').strip()
            elem_class = element.get('class', '').strip()
            direct_text = self._clean_text(element.text)
            
            # 筛选逻辑：只保留有意义的节点
            is_significant = False
            if elem_id: is_significant = True
            elif direct_text: is_significant = True
            elif tag == 'img': is_significant = True
            elif elem_class: is_significant = True # 只要有 Class 就保留，防止漏掉容器
            elif tag == 'a' and element.get('href'): is_significant = True
            
            if not is_significant: continue

            try:
                xpath = tree.getroottree().getpath(element)
            except ValueError: continue

            # --- 格式化输出 (修复 Class 显示问题) ---
            tag_repr = tag
            if elem_id: tag_repr += f"#{elem_id}"
            
            if elem_class:
                # 【关键修复】: 显示所有 Class，用点号连接，不再截断！
                # 将 "bt_img mi_ne_kd newindex" 转换为 ".bt_img.mi_ne_kd.newindex"
                classes = elem_class.split()
                # 过滤掉极其通用的无意义类名（可选）
                # classes = [c for c in classes if c not in ['on', 'active', 'clearfix']]
                tag_repr += "." + ".".join(classes)
                
                # 标记轮播图
                if any(kw in elem_class.lower() for kw in self.CAROUSEL_KEYWORDS):
                     tag_repr += " [CAROUSEL?]"

            content_preview = ""
            if direct_text: content_preview = f"Txt:{direct_text[:max_text_len]}"
            elif tag == 'a': content_preview = f"Href:{element.get('href', '')[:30]}"
            elif tag == 'img': content_preview = f"Img:{element.get('src', '')[:30]}"

            # 这里的 <60 是为了给长类名留足空间
            line = f"{node_count:03d} | {tag_repr:<60} | {content_preview:<30} | {xpath}"
            skeleton_lines.append(line)
            node_count += 1
            
        return "\n".join(skeleton_lines)

    def extract_by_xpath(self, html_content: str, rules: Dict[str, str]) -> List[Dict]:
        """本地执行 XPath 提取"""
        if not html_content or not rules: return []
        try:
            tree = html.fromstring(html_content)
        except Exception: return []
        
        container_xpath = rules.get("container")
        field_rules = rules.get("fields", {})
        
        if not container_xpath: return []
        results = []
        
        try:
            containers = tree.xpath(container_xpath)
        except Exception as e:
            print(f"❌ Invalid Container XPath: {e}")
            return []

        if not containers:
            # 增加一些 Debug 信息
            print(f"⚠️ XPath 未找到任何容器: {container_xpath}")
            return []

        for item in containers:
            data = {}
            has_data = False
            for key, sub_xpath in field_rules.items():
                try:
                    # 容错：强制相对路径
                    if sub_xpath.startswith("/"): sub_xpath = "." + sub_xpath
                    if not sub_xpath.startswith("."): sub_xpath = "./" + sub_xpath
                    
                    nodes = item.xpath(sub_xpath)
                    if nodes:
                        if isinstance(nodes[0], html.HtmlElement):
                            data[key] = self._clean_text(nodes[0].text_content())
                        else:
                            data[key] = self._clean_text(str(nodes[0]))
                        if data[key]: 
                            has_data = True
                    else:
                        data[key] = None
                except Exception:
                    data[key] = None
            
            if has_data:
                results.append(data)
                
        return results

dom_analyzer = DOMAnalyzer()
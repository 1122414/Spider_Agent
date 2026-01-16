DOM_SKELETON_JS = """
(function() {
    window.__dom_result = null;
    window.__dom_status = 'pending';

    try {
        console.time("DOM_Analysis");
        console.log("🚀 [Visual Nerve] 正在扫描页面结构...");

        // ================= 配置区 =================
        const CONFIG = {
            MAX_DEPTH: 50,          // 深度限制，防止栈溢出
            MAX_TEXT_LEN: 80,       // 文本截断长度
            LIST_HEAD_COUNT: 5,     // 列表保留头部数量
            LIST_TAIL_COUNT: 1,     // 列表保留尾部数量
            ATTRIBUTES_TO_KEEP: ['href', 'src', 'title', 'placeholder', 'type', 'aria-label', 'role', 'data-id'] // 关键属性白名单
        };

        // ================= 核心工具函数 =================
        
        // 1. 生成唯一的 XPath (绝对路径)
        function getXPath(element) {
            if (element.id && element.id.match(/^[a-zA-Z][a-zA-Z0-9_-]*$/)) {
                // 如果 ID 看起来很干净且唯一，优先使用 ID (缩短路径)
                // 排除自动生成的乱码 ID
                return '//*[@id="' + element.id + '"]';
            }
            if (element === document.body) return '/html/body';

            let ix = 0;
            if (!element.parentNode) return ''; // 游离节点
            
            let siblings = element.parentNode.childNodes;
            for (let i = 0; i < siblings.length; i++) {
                let sibling = siblings[i];
                if (sibling === element) {
                    let parentPath = getXPath(element.parentNode);
                    return parentPath + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                }
                if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                    ix++;
                }
            }
        }

        // 2. 判断元素是否可见 (优化版，避免过多重排)
        function isVisible(elem) {
            // 排除显式隐藏
            if (elem.style.display === 'none' || elem.style.visibility === 'hidden') return false;
            
            // 某些关键标签即使不可见也要保留 (如 hidden inputs 用于传参)
            if (elem.tagName === 'INPUT' && elem.type === 'hidden') return true;

            // 获取计算样式 (开销较大，仅对非文本节点检查)
            // 这里为了性能，假设如果没有宽高的块级元素且没子节点可能是不可见的
            // 但为了保险起见，全栈爬虫建议还是保留结构，依靠 LLM 判断
            return true; 
        }

        // 3. 递归遍历 DOM 生成 JSON
        function traverse(node, depth) {
            if (depth > CONFIG.MAX_DEPTH) return null;
            if (!node) return null;

            // --- 过滤层 ---
            // 1. 标签过滤
            const skipTags = ['SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'PATH', 'HEAD', 'META', 'LINK', 'IFRAME', 'BR', 'HR', 'WBR'];
            if (skipTags.includes(node.tagName)) return null;

            // 2. 节点类型过滤 (只处理元素和非空文本)
            if (node.nodeType !== 1) return null;

            // 3. 可见性过滤 (简单判断，过于复杂的判断会拖慢 JS 执行)
            // 只有当元素完全透明或 display:none 时才跳过
            // 注意：不要用 getComputedStyle 遍历全站，太慢。这里只做基础判断。
            
            // --- 数据提取层 ---
            let info = {
                t: node.tagName.toLowerCase(), // tag
                x: getXPath(node)              // xpath
            };

            // 提取关键属性
            if (node.id) info.id = node.id;
            if (node.className && typeof node.className === 'string' && node.className.trim()) {
                info.c = node.className.trim(); // class
            }
            
            CONFIG.ATTRIBUTES_TO_KEEP.forEach(attr => {
                let val = node.getAttribute(attr);
                if (val) {
                    // 截断过长的 URL
                    if (val.length > 100 && (attr === 'href' || attr === 'src')) val = val.substring(0, 100) + '...';
                    info[attr] = val;
                }
            });

            // 提取自身直接包含的文本 (不含子元素文本)
            let directText = "";
            node.childNodes.forEach(child => {
                if (child.nodeType === 3) { // Text Node
                    let txt = child.textContent.trim();
                    if (txt) directText += txt + " ";
                }
            });
            if (directText.trim()) {
                info.txt = directText.trim().substring(0, CONFIG.MAX_TEXT_LEN);
            }

            // --- 递归子节点 (核心改进：列表采样) ---
            let children = Array.from(node.children);
            
            if (children.length > 0) {
                info.kids = [];
                
                // 判断是否为列表结构：子元素数量多且标签名相同
                let isList = children.length > 8; 
                
                if (isList) {
                    // 采样模式：头几项 + 尾几项
                    let head = children.slice(0, CONFIG.LIST_HEAD_COUNT);
                    let tail = children.slice(children.length - CONFIG.LIST_TAIL_COUNT);
                    
                    // 处理头部
                    head.forEach(child => {
                        let c = traverse(child, depth + 1);
                        if (c) info.kids.push(c);
                    });
                    
                    // 插入省略标记，告诉 LLM 这里跳过了多少项
                    info.kids.push({
                        t: "skipped_items",
                        count: children.length - head.length - tail.length,
                        desc: `... ${children.length - head.length - tail.length} more items ...`
                    });

                    // 处理尾部
                    tail.forEach(child => {
                        let c = traverse(child, depth + 1);
                        if (c) info.kids.push(c);
                    });

                } else {
                    // 非列表，完整遍历
                    children.forEach(child => {
                        let c = traverse(child, depth + 1);
                        if (c) info.kids.push(c);
                    });
                }
            }

            // --- 剪枝层 (最后防线) ---
            // 如果一个节点既没有 ID/Class/Text/Attributes，也没有子节点，那它就是废节点
            let hasAttr = Object.keys(info).some(k => CONFIG.ATTRIBUTES_TO_KEEP.includes(k));
            if (!info.id && !info.c && !info.txt && !hasAttr && (!info.kids || info.kids.length === 0)) {
                // 特殊放行：INPUT 和 IMG 即使没内容也要保留
                const selfClosing = ['input', 'img', 'button', 'select', 'textarea'];
                if (!selfClosing.includes(info.t)) return null;
            }

            return info;
        }

        // ================= 执行入口 =================
        // 优先寻找主要内容容器，减少 Header/Footer 干扰
        // 策略：如果找到了 #content 或 main 标签，优先以此为根，否则用 body
        let root = document.getElementById('content') || 
                   document.querySelector('main') || 
                   document.querySelector('.container') ||
                   document.body;
                   
        // 兜底：如果找到的 root 内容太少（可能是个空壳），还是回退到 body
        if (root.innerText.length < 50 && root !== document.body) {
            root = document.body;
        }
        
        console.log(`🎯 锁定分析根节点: <${root.tagName} class="${root.className}" id="${root.id}">`);

        let result = traverse(root, 0);

        if (!result) {
            window.__dom_result = JSON.stringify({error: "Empty DOM"});
            window.__dom_status = 'error';
        } else {
            // 添加元数据，告诉 Python 这里的根节点不是 HTML，要注意 XPath 拼接
            result.is_fragment = (root !== document.body && root !== document.documentElement);
            window.__dom_result = JSON.stringify(result);
            window.__dom_status = 'success';
        }
        
        console.timeEnd("DOM_Analysis");
        console.log("✅ 视觉神经信号已生成 (长度: " + window.__dom_result.length + ")");

    } catch (e) {
        console.error("❌ 视觉神经崩溃:", e);
        window.__dom_result = JSON.stringify({error: e.toString()});
        window.__dom_status = 'error';
    }
})();
"""
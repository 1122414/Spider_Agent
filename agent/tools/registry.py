from typing import Dict, Any, Callable, List

class ToolRegistry:
    def __init__(self):
        # 初始化为空字典，没有任何预设工具
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, str] = {}
        self.history: List[Dict] = []

    def register_tool(self, tool_name: str, func: Callable, description: str):
        """注册新工具"""
        print(f"[Registry] Registering tool: {tool_name}")
        self._tools[tool_name] = func
        self._tool_descriptions[tool_name] = description

    def get_available_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_description_prompt(self) -> str:
        """生成给 LLM 看的工具描述列表"""
        prompt = ""
        for name, desc in self._tool_descriptions.items():
            prompt += f"- {name}: {desc}\n"
        return prompt

    def get_recent_history(self, limit: int = 3) -> List[Dict]:
        return self.history[-limit:]

    def add_to_history(self, tool_name, params, result):
        self.history.append({
            "tool": tool_name,
            "params": params,
            "result": str(result)[:500]  # 截断过长的结果
        })

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """根据 tool_name 查找并执行对应的函数"""
        print(f"正在执行工具: {tool_name}，参数: {parameters}")

        # 1. 检查工具是否存在
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"}
        
        # 2. 获取函数对象
        tool_func = self._tools[tool_name]
        
        # 3. 执行函数
        try:
            # 使用 **parameters 将字典解包为关键字参数
            result = tool_func(**parameters)
            # 等同于调用：
            # result = sync_playwright_fetch(url="...", target=[...])
            return result
        except TypeError as e:
            return {"error": f"Parameter mismatch for tool '{tool_name}': {str(e)}"}
        except Exception as e:
            return {"error": f"Execution failed for '{tool_name}': {str(e)}"}

# 全局单例
tool_registry = ToolRegistry()
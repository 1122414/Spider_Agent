from typing import Dict, Any, Callable, List

class ToolRegistry:
    def __init__(self):
        # 初始化工具字典
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, str] = {}
        self.history: List[Dict] = []
        
        # 【新增】暂存上一次工具执行的完整结果 (数据总线)
        # 用于在工具之间隐式传递大数据，无需经过 LLM
        self.last_execution_result: Any = None

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

    def get_recent_history(self, limit: int = 5) -> List[Dict]:
        """获取最近的历史记录供 Prompt 使用"""
        return self.history[-limit:]

    def add_to_history(self, tool_name, params, result):
        """
        记录工具执行历史
        【关键】如果结果太长（例如爬虫数据），进行截断！防止 Prompt 爆炸
        """
        result_str = str(result)
        if len(result_str) > 2000:
            display_result = result_str[:2000] + f"\n... (剩余 {len(result_str)-2000} 字符已省略，完整数据已缓存) ..."
        else:
            display_result = result_str

        self.history.append({
            "tool": tool_name,
            "params": params,
            "result": display_result 
        })

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """根据 tool_name 查找并执行对应的函数"""
        print(f"⚙️ 正在执行工具: {tool_name}，参数: {parameters}")

        # 1. 检查工具是否存在
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"}
        
        # 2. 获取函数对象
        tool_func = self._tools[tool_name]
        
        # 3. 执行函数
        try:
            # 使用 **parameters 将字典解包为关键字参数
            result = tool_func(**parameters)
            
            # 【关键】将结果存入内存，供下一个工具（如 save_tool）自动获取
            self.last_execution_result = result
            
            return result
        except TypeError as e:
            return {"error": f"Parameter mismatch for tool '{tool_name}': {str(e)}"}
        except Exception as e:
            return {"error": f"Execution failed for '{tool_name}': {str(e)}"}

# 全局单例
tool_registry = ToolRegistry()
from typing import Dict, Any, Callable, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, str] = {}
        self.history: List[Dict] = []
        self.last_execution_result: Any = None

    def register_tool(self, tool_name: str, func: Callable, description: str):
        print(f"[Registry] Registering tool: {tool_name}")
        self._tools[tool_name] = func
        self._tool_descriptions[tool_name] = description

    def get_available_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_description_prompt(self) -> str:
        prompt = ""
        for name, desc in self._tool_descriptions.items():
            prompt += f"- {name}: {desc}\n"
        return prompt

    def get_recent_history(self, limit: int = 5) -> List[Dict]:
        return self.history[-limit:]

    def add_to_history(self, tool_name, params, result):
        result_str = str(result)
        if len(result_str) > 2000:
            display_result = result_str[:2000] + f"\n... (剩余 {len(result_str)-2000} 字符已省略) ..."
        else:
            display_result = result_str

        self.history.append({
            "tool": tool_name,
            "params": params,
            "result": display_result 
        })

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        print(f"⚙️ 正在执行工具: {tool_name}，参数: {parameters}")

        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found."}
        
        tool_func = self._tools[tool_name]
        
        try:
            result = tool_func(**parameters)
            
            # 【保护逻辑】只有非保存工具才能更新数据缓存
            if not tool_name.startswith("save_"):
                self.last_execution_result = result
            
            return result
        except TypeError as e:
            return {"error": f"Parameter mismatch: {str(e)}"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

tool_registry = ToolRegistry()
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agent.tools.registry import tool_registry
from agent.prompt_template import TOOLS_USED_SYSTEM_PROMPT


class DecisionEngine:
    """
    Agent决策引擎 - 实现ReAct模式的思考-行动-观察循环
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.history = []
    
    def think_and_act(self, task: str) -> Dict[str, Any]:
        """ReAct决策循环的核心实现"""
        
        # 获取可用工具列表
        available_tools = tool_registry.get_available_tools()
        tools_list = tool_registry.get_tool_description_prompt()
        recent_history = tool_registry.get_recent_history(3)
        
        # 使用PromptTemplate格式化提示词
        prompt = PromptTemplate.from_template(TOOLS_USED_SYSTEM_PROMPT)
        
        # 调用LLM进行决策
        response = self.llm.invoke([
            {"role": "system", "content": prompt.format(
                tools_list=tools_list,
                recent_history=recent_history,
                task=task
            )},
            {"role": "user", "content": f"请分析任务并决定下一步行动"}])
        
        try:
            decision = json.loads(response.content)
            
            # 验证决策格式
            if "action" not in decision:
                return {"error": "Invalid decision format"}
            
            if decision["action"] == "stop":
                return {
                    "status": "completed",
                    "final_thought": decision.get("thought", ""),
                    "history": self.history
                }
            
            # 执行工具
            if decision["action"] == "next" and "tool_name" in decision:
                tool_result = tool_registry.execute_tool(
                    decision["tool_name"],
                    decision.get("parameters", {})
                )
                
                # 记录执行历史
                tool_registry.add_to_history(
                    decision["tool_name"],
                    decision.get("parameters", {}),
                    tool_result
                )
                
                return {
                    "status": "in_progress",
                    "thought": decision.get("thought", ""),
                    "tool_used": decision["tool_name"],
                    "tool_result": tool_result
                }
            else:
                return {"error": "Invalid action or missing tool_name"}
                
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse decision: {str(e)}"}
    
    def is_task_completed(self, observation: Dict, original_task: str) -> bool:
        """判断任务是否完成"""
        # 检查是否有错误
        if observation.get("error"):
            return False
            
        return True


# 全局决策引擎实例
decision_engine = None

def init_decision_engine(llm: ChatOpenAI):
    """初始化决策引擎"""
    global decision_engine
    decision_engine = DecisionEngine(llm)
    return decision_engine

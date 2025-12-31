import json
from typing import Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.tools.registry import tool_registry
from agent.prompt_template import build_system_prompt

# =============================================================================
# 1. 定义结构化输出的 Schema (这是稳定性的核心)
# =============================================================================
class AgentDecisionSchema(BaseModel):
    """Agent 的决策结构，包含思考过程和行动指令"""
    
    thought: str = Field(
        default="",
        description="你的思考过程：分析当前状态、评估上一步结果、决定下一步计划。"
    )
    action: Literal["next", "stop"] = Field(
        ..., 
        description="决策行动：'next' 表示调用工具继续执行，'stop' 表示任务已完成或无法继续。"
    )
    tool_name: Optional[str] = Field(
        None, 
        description="要调用的工具名称。如果 action 是 'stop'，此字段可留空。"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="工具调用的参数字典。如果 action 是 'stop'，此字段可留空。"
    )

class DecisionEngine:
    """
    Agent决策引擎 (Function Calling 版)
    使用 LLM 的 Tool Calling 能力来保证输出格式的绝对稳定。
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 绑定结构化输出 Schema
        # 这一步会让 LLM 强制输出符合 AgentDecisionSchema 的 JSON
        self.structured_llm = self.llm.with_structured_output(AgentDecisionSchema)
        self.max_steps = 10
    
    def think_and_act(self, task: str) -> Dict[str, Any]:
        """ReAct决策循环的核心实现"""
        print(f"🎯 收到新任务: {task}")
        
        # 每次新任务清空历史和缓存
        tool_registry.history = []
        tool_registry.last_execution_result = None
        
        step_count = 0
        
        while step_count < self.max_steps:
            step_count += 1
            print(f"\n🔄 [Step {step_count}] Agent 正在思考 (Structured)...")
            
            # 1. 准备上下文
            # 注意：使用 Structured Output 后，我们依然需要 System Prompt 来描述工具功能，
            # 但不再需要费力地去教 LLM "如何输出 JSON"，因为它已经知道了。
            tools_desc = tool_registry.get_tool_description_prompt()
            
            # 获取最近执行历史
            raw_history = tool_registry.get_recent_history(5)
            # 将历史转换为易读的字符串格式，供 LLM 参考
            recent_history_str = json.dumps(raw_history, ensure_ascii=False, indent=2)
            
            # 2. 构造 Prompt
            system_prompt_str = build_system_prompt(
                tools_desc=tools_desc,
                recent_history_str=recent_history_str,
                task=task
            )
            
            # 3. 调用 LLM (使用结构化模式)
            try:
                # 构造消息列表
                messages = [
                    ("system", system_prompt_str),
                    ("user", f"当前任务状态如上。请根据 {task} 进行下一步决策。")
                ]
                
                # invoke 返回的直接是 AgentDecisionSchema 的实例对象
                decision_obj: AgentDecisionSchema = self.structured_llm.invoke(messages)
                
                # 转为字典方便处理  model对象的函数  继承BaseModel
                decision = decision_obj.model_dump()
                
            except Exception as e:
                print(f"❌ LLM 调用或解析失败: {e}")
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": f"Decision failed: {str(e)}"}
            
            # 4. 执行决策逻辑
            action = decision.get("action")
            thought = decision.get("thought", "")
            tool_name = decision.get("tool_name")
            params = decision.get("parameters", {})

            print(f"💡 Thought: {thought}")
            
            # --- 分支 A: 停止/完成 ---
            if action == "stop":
                print("🏁 任务完成或停止。")
                return {
                    "status": "completed",
                    "final_thought": thought,
                    "history": tool_registry.history
                }
            
            # --- 分支 B: 继续执行工具 ---
            elif action == "next":
                if not tool_name:
                    print("⚠️ 警告: LLM 决定继续，但未提供工具名称，强制重试...")
                    continue

                # 执行工具
                result = tool_registry.execute_tool(tool_name, params)
                
                # 将结果写入历史，供下一轮思考使用
                tool_registry.add_to_history(tool_name, params, result)
                
            else:
                return {"status": "error", "message": f"Invalid action: {action}"}
                
        return {"status": "timeout", "message": "达到最大步数限制"}

# 全局单例管理
decision_engine = None

def init_decision_engine(llm: ChatOpenAI):
    global decision_engine
    decision_engine = DecisionEngine(llm)
    return decision_engine
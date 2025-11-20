import json
import re
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agent.tools.registry import tool_registry
from agent.prompt_template import TOOLS_USED_SYSTEM_PROMPT

class DecisionEngine:
    """
    Agent决策引擎 - 实现ReAct模式的多步思考-行动循环
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 设置最大步数防止死循环
        self.max_steps = 10
    
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        【新增】鲁棒的 JSON 解析器
        能够处理 LLM 返回的各种非标准 JSON 格式（Markdown 包裹、前后有废话等）
        """
        text = text.strip()
        
        # 策略 1: 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 2: 去除常见的 Markdown 代码块标记
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 策略 3: 使用正则表达式提取第一个 JSON 对象 {...}
        # [\s\S]* 匹配任意字符包括换行
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                potential_json = match.group(0)
                return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
        
        # 如果都失败了，抛出异常供上层捕获
        raise ValueError(f"无法从 LLM 响应中提取有效 JSON。响应预览: {text[:100]}...")

    def think_and_act(self, task: str) -> Dict[str, Any]:
        """ReAct决策循环的核心实现"""
        print(f"🎯 收到新任务: {task}")
        
        # 每次新任务清空历史和缓存，防止上下文污染
        tool_registry.history = []
        tool_registry.last_execution_result = None
        
        step_count = 0
        
        while step_count < self.max_steps:
            step_count += 1
            print(f"\n🔄 [Step {step_count}] Agent 正在思考...")
            
            # 1. 准备上下文
            available_tools = tool_registry.get_available_tools()
            tools_list = tool_registry.get_tool_description_prompt()
            
            # 获取历史记录（registry 中已自动截断过长内容）
            raw_history = tool_registry.get_recent_history(5)
            recent_history_str = json.dumps(raw_history, ensure_ascii=False, indent=2)
            
            # 2. 构造 Prompt
            system_prompt = TOOLS_USED_SYSTEM_PROMPT.replace(
                "{tools_list}", tools_list
            ).replace(
                "{recent_history}", recent_history_str
            ).replace(
                "{task}", task
            )
            
            # 3. 调用 LLM 进行决策
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请根据当前状态和任务目标，严格以 JSON 格式决定下一步行动 (next) 或 结束任务 (stop)。"}
                ])
                
                content = response.content.strip()
                
                # 使用增强的解析器
                decision = self._parse_json_safely(content)
                
            except Exception as e:
                print(f"❌ 决策解析失败: {e}")
                # 可以选择重试，或者直接报错返回
                return {"status": "error", "message": f"Decision parse failed: {str(e)}"}
            
            # 4. 执行决策
            action = decision.get("action")
            thought = decision.get("thought", "")
            print(f"💡 Thought: {thought}")
            
            if action == "stop":
                print("🏁 任务完成。")
                return {
                    "status": "completed",
                    "final_thought": thought,
                    "history": tool_registry.history
                }
            
            elif action == "next":
                tool_name = decision.get("tool_name")
                if not tool_name:
                    return {"status": "error", "message": "Missing tool_name in next action"}

                params = decision.get("parameters", {})
                
                # 执行工具
                result = tool_registry.execute_tool(tool_name, params)
                
                # 记录历史 (Registry 会自动截断并缓存结果)
                tool_registry.add_to_history(tool_name, params, result)
                
                # 循环继续，Agent 将看到这一步的执行结果...
            else:
                return {"status": "error", "message": f"Invalid action: {action}"}
                
        return {"status": "timeout", "message": "达到最大步数限制"}

# 全局决策引擎实例
decision_engine = None

def init_decision_engine(llm: ChatOpenAI):
    """初始化决策引擎"""
    global decision_engine
    decision_engine = DecisionEngine(llm)
    return decision_engine
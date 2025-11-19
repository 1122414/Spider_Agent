# AutoCrawlerAgent 升级方案

## 当前项目 → 真正 Agent 的转型路径

### 阶段一：基础 Agent 架构（Agent 1.0）

#### 1. 工具注册系统

```python
# 新增 agent/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, description: str, function: callable):
        self.tools[name] = {
            "description": description,
            "function": function
        }

    def get_available_tools(self) -> dict:
        return self.tools
```

#### 2. 决策循环实现（ReAct 模式）

```python
# 修改 agent/langchain_agent.py
def run_agent_task_v2(nl_text: str):
    """真正的Agent工作流程"""
    tools = tool_registry.get_available_tools()

    # Agent思考-行动-观察循环
    max_iterations = 5
    for i in range(max_iterations):
        # 1. 思考：分析当前状态，决定下一步
        thought = agent_think(nl_text, tools, execution_history)

        # 2. 行动：执行选择的工具
        if thought.get("action") == "stop":
            break

        # 3. 观察：收集执行结果
        observation = execute_tool(thought["tool_name"], thought["parameters"])

        # 4. 评估：判断任务是否完成
        if is_task_completed(observation, nl_text):
            break
```

#### 3. 工具注册实现

```python
# 在 app.py 中初始化
tool_registry = ToolRegistry()
tool_registry.register_tool(
    name="web_crawler",
    description="爬取网页内容并提取结构化数据",
    function=sync_playwright_fetch
)

tool_registry.register_tool(
    name="knowledge_search",
    description="在已有知识库中搜索相关信息",
    function=qa_interaction
)
```

### 阶段二：高级 Agent 功能（Agent 2.0）

#### 1. 多工具协同

```python
# Agent可以自主决定：
# - 先搜索知识库获取背景信息
# - 再爬取相关网页补充数据
# - 最后进行综合分析
```

## 具体实现步骤

### 第一步：创建工具注册系统

```python
# agent/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.execution_history = []

    def register_tool(self, name: str, description: str, function: callable):
        self.tools[name] = {
            "description": description,
            "function": function
        }

    def execute_tool(self, tool_name: str, parameters: dict):
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        try:
            result = self.tools[tool_name]["function"](**parameters)
        except Exception as e:
            result = {"error": str(e)}

        self.execution_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result
        })
        return result
```

### 第二步：实现决策循环

```python
# agent/decision_engine.py
class DecisionEngine:
    def __init__(self, tool_registry, llm):
        self.tool_registry = tool_registry
        self.llm = llm
        self.history = []

    def think_and_act(self, task: str, context: dict = None):
        """ReAct模式的核心实现"""
        current_state = {
            "task": task,
            "available_tools": list(tool_registry.tools.keys()),
        "history": self.history
        }

        # 让AI分析当前状态并决定下一步
        decision_prompt = """
        当前任务: {task}
        可用工具: {tools}
        执行历史: {history}

        请分析并返回JSON:
        {
            "thought": "你的思考过程",
            "action": "next|stop",
            "tool_name": "工具名称",
            "parameters": {参数字典}
        }
        """

        response = self.llm.invoke(decision_prompt)
        return json.loads(response.content)
```

### 第三步：增强提示工程

```python
# 修改 agent/prompt_template.py
AGENT_SYSTEM_PROMPT = """
你是一个智能网页爬虫Agent，可以自主使用工具完成任务。

可用工具:
{tools_list}

思考原则:
1. 分析任务需求，确定需要哪些信息
2. 选择合适的工具和参数
3. 分析执行结果，判断是否需要继续

输出必须是合法JSON，包含thought、action、tool_name、parameters字段。
"""
```

### 第四步：集成与测试

```python
# 新的主程序 agent/main_agent.py
class AutoCrawlerAgent:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.decision_engine = DecisionEngine(self.tool_registry, chat)

    def run(self, task: str):
        return self.decision_engine.think_and_act(task)
```

## 预期效果

### 升级后的能力对比

| 功能       | 当前   | 升级后      |
| ---------- | ------ | ----------- |
| 工具选择   | 硬编码 | AI 自主选择 |
| 工作流     | 固定   | 动态规划    |
| 任务评估   | 无     | 自主判断    |
| 多步骤任务 | 不支持 | 完全支持    |
| 错误处理   | 基础   | 智能恢复    |

## 实施优先级

### 高优先级（立即开始）

1. 工具注册系统实现
2. 基础决策循环
3. 执行历史跟踪

### 中优先级（一周内）

1. 多工具协同工作
2. 结果质量评估
3. 自动优化策略

### 低优先级（后续优化）

1. 长期记忆存储
2. 技能学习机制
3. 复杂任务分解

## 技术依赖

- LangChain Agent 框架
- ReAct 决策模式
- 工具调用标准化
- 状态管理机制

## 总结

通过以上四个阶段的升级，你的项目将从"AI 辅助工具"转变为真正的"自主 AI Agent"，具备：

- ✅ 自主工具选择
- ✅ 动态工作规划
- ✅ 目标导向执行
- ✅ 智能错误恢复

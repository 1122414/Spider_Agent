# AutoCrawlerAgent V2 - 智能网页爬虫代理系统

一个基于 LangChain 和 Playwright 的智能网页爬虫代理，支持多层级递归抓取、数据持久化和 RAG 知识库问答。

## 🚀 项目概述

AutoCrawlerAgent V2 是一个先进的智能网页数据采集系统，具有以下核心特性：

- **自然语言交互**：用户可通过自然语言描述爬取需求
- **多层级递归抓取**：支持列表页 → 详情页 → 更多详情页的深度钻取
- **持久化浏览器会话**：自动保持登录状态和 Cookies
- **智能数据提取**：基于 LLM 的内容理解和结构化
- **RAG 知识库集成**：将抓取数据存入向量数据库，支持智能问答
- **抗反爬能力**：模拟真实浏览器行为，绕过常见反爬机制
- **多格式数据导出**：支持 JSON、CSV、PostgreSQL、Milvus 向量库

## 📋 核心功能

### 1. 智能爬取引擎

- **基础爬虫**：单页面内容提取，支持滚动触发懒加载
- **深度爬虫**：多层级递归抓取，自动翻页处理
- **数据清洗与结构化**：自动去除广告、导航等噪音信息
- **字段映射**：智能识别用户需求字段，自动生成结构化数据
- **链接补全**：自动处理相对路径，生成完整 URL

### 2. 决策引擎 (ReAct 模式)

- **多步思考-行动循环**：自主规划执行路径
- **工具调度**：动态选择和调用不同爬取和保存工具

### 3. 多格式数据保存

- **JSON 文件**：结构化数据存储
- **CSV 表格**：便于数据分析和处理
- **PostgreSQL 数据库**：关系型数据存储
- **Milvus 向量库**：用于 RAG 问答的知识库

### 4. RAG 知识库问答

- **基于上下文的智能回答**：利用向量检索提供准确答案
- **动态检索策略**：根据问题类型自动调整检索数量

## 🏗️ 系统架构

```
AutoCrawlerAgent V2
├── app_agent.py              # 主程序入口
├── config.py                  # 统一配置管理
├── agent/                      # 智能代理核心模块
│   ├── decision_engine.py      # ReAct决策引擎
│   ├── prompt_template.py      # 系统提示词模板
└── agent/tools/               # 工具集
    ├── registry.py             # 工具注册中心
    ├── crawl_tool.py            # 爬虫工具
    ├── extractor_agent.py        # 数据提取代理
│   ├── ingest_tool.py           # RAG入库工具
│   ├── save_tool.py             # 数据保存工具
    └── extractor_agent.py       # 内容提取引擎
```

### 模块详解

#### 1. 决策引擎 ([`agent/decision_engine.py`](agent/decision_engine.py))

- 实现 ReAct 模式的多步决策循环
- 自动规划工具执行顺序
- 防止死循环的最大步数保护

#### 2. 爬虫工具 ([`agent/tools/crawl_tool.py`](agent/tools/crawl_tool.py))

- 支持单页面和多层级抓取
- 浏览器会话持久化，保持登录状态
- 智能滚动触发懒加载内容

#### 3. 数据提取代理 ([`agent/tools/extractor_agent.py`](agent/tools/extractor_agent.py))

- 基于 LLM 的智能内容理解
- 分块处理大型网页内容
- 自动去重和数据清洗

#### 4. RAG 知识库 ([`rag/`](rag/))

- 向量化存储：支持 OpenAI 和 Ollama 两种嵌入模型
- 动态检索策略：根据问题类型调整 Top-K 值

## 🔧 技术栈

### 核心框架

- **LangChain 0.3.8**：智能代理框架
- **Playwright**：现代化浏览器自动化
- **BeautifulSoup4**：HTML 解析
- **ChromaDB** / **Milvus**：向量数据库
- **OpenAI API** / **Ollama**：LLM 和嵌入模型

### 主要依赖

```python
# 核心框架
langchain==1.0.8
langchain-openai==1.0.3

# 浏览器自动化
playwright>=1.48.0

# 向量数据库
chromadb>=0.4.0
langchain-chroma>=0.1.0

# 嵌入模型
openai>=1.58.1

# 数据库支持
psycopg2  # PostgreSQL连接
pymilvus  # Milvus向量库
```

### 数据存储

- **JSON/CSV 文件**：本地数据存储
- **PostgreSQL**：关系型数据库
- **Milvus**：高性能向量数据库

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install
```

### 2. 配置设置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下关键参数：

```env
# 魔搭模型配置
MODA_OPENAI_API_KEY="your_openai_api_key"
MODA_OPENAI_BASE_URL="https://api.moda.com/v1"

# 向量数据库配置
MILVUS_URI="http://localhost:19530"
COLLECTION_NAME="spider_knowledge_base"

# 本地 Ollama 配置 (可选)
OPENAI_OLLAMA_BASE_URL="http://localhost:11434"

# 嵌入模型选择
EMBEDDING_TYPE="api"  # 或 "local" 使用 Ollama
```

### 3. 启动系统

```bash
python app_agent.py
```

## 💡 使用示例

### 基础爬取

```
👤 User > 帮我抓取 https://example.com 网站的标题和描述
```

### 多层级深度爬取

```
👤 User > 抓取电影网站，第一层获取电影列表，第二层获取详细信息
```

### RAG 问答模式

```
👤 User > qa 这个网站上有哪些电影？
```

### 复杂任务示例

```
👤 User > 抓取携程网站上的西安旅游攻略，保存到知识库
```

## 🛠️ 工具说明

### 已注册工具

#### 1. web_crawler

- **功能**：单页面基础爬取工具
- **参数**：
  - `url`：目标网址
  - `target`：需要提取的字段列表
- **特性**：持久化浏览器会话，抗反爬能力

#### 2. hierarchical_crawler

- **功能**：多层级递归深度抓取
- **参数**：
  - `url`：起始网址
  - `crawl_scopes`：二维数组，定义每一层的抓取目标
- **支持**：自动翻页、登录状态保持

#### 3. 数据保存工具

- `save_to_json`：保存为 JSON 文件
- `save_to_csv`：保存为 CSV 表格
- `save_to_postdb`：保存到 PostgreSQL 数据库
- `save_to_knowledge_base`：存入 Milvus 向量知识库

## 🔍 RAG 知识库问答

系统支持基于已存入知识库的智能问答：

```bash
# 直接进入RAG问答模式
qa 告诉我知识库里有哪些内容？
```

## 📁 项目结构

```
.
├── app_agent.py                 # 主程序入口
├── config.py                     # 统一配置管理
├── requirements.txt              # 项目依赖
├── Readme.md                    # 项目说明文档
├── .env                         # 环境变量配置
├── agent/
│   ├── __init__.py
│   ├── decision_engine.py      # ReAct决策引擎
│   ├── prompt_template.py       # 系统提示词模板
│   └── tools/
│       ├── __init__.py
│       ├── registry.py          # 工具注册中心
│       ├── crawl_tool.py        # 爬虫工具
│       ├── extractor_agent.py    # 数据提取代理
│       ├── ingest_tool.py        # RAG入库工具
│       └── save_tool.py         # 数据保存工具
├── rag/
│   ├── __init__.py
│   ├── vectorstore.py           # 向量存储管理
│       └── retriever_qa.py       # RAG问答系统
├── utils/
│   ├── helpers.py                # 辅助函数
│   └── text_splitter.py         # 文本切分工具
├── output/                       # 数据输出目录
├── chroma_data/                   # ChromaDB 数据存储
└── item_md/                      # 项目文档
```

## ⚙️ 配置说明

### 模型配置

- **OpenAI API**：支持 GPT-4o-mini 等模型
  ├── abandon/ # 废弃代码目录
  └── test/ # 测试代码

````

## 🎯 核心特性

### 1. 智能决策
- 基于ReAct模式的自主决策
- 动态工具选择和参数配置

### 2. 持久化爬取
- 浏览器数据保存在 `./browser_data` 目录
- 自动复用会话，保持登录状态
- 模拟人类浏览行为，绕过反爬机制

### 3. 抗反爬能力
- 随机等待时间
- 模拟滚动行为
- 真实浏览器指纹

### 4. 可扩展架构
- 模块化工具设计
- 易于添加新功能
- 支持多种数据源和目标

## 🔗 依赖管理

项目使用标准的Python依赖管理：

```bash
# 安装所有依赖
pip install -r requirements.txt

# 开发依赖
pip install pytest flake8 black
````

## 📝 开发说明

### 代码规范

- 遵循 PEP 8 编码规范
- 使用类型注解提高代码可读性

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 📄 许可证

[请在此处添加许可证信息]

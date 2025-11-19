# AutoCrawlerAgent

功能：自然语言输入 -> LangChain 解析任务 -> DrissionPage 执行爬取 -> 将结果分块 & embedding -> 存入 Qdrant 向量库 -> 支持 RAG 问答

## 快速开始

1. 复制 `.env.example` 为 `.env` 并填写 `OPENAI_API_KEY` 等配置。
2. chroma run --path ./chroma_data --host 0.0.0.0 --port 7070

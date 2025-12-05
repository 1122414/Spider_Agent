import os
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 导入配置
from config import (
    MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL
)
from agent.prompt_template import QUERY_ANALYZER_PROMPT

# 定义元数据过滤的 Schema
class MetadataFilter(BaseModel):
    """
    从用户问题中提取的元数据过滤条件。
    """
    # Category 字段：用于大类过滤
    category: Optional[str] = Field(
        None,
        description="""
        用户查询涉及的【内容大类】或【领域】。
        常见值包括：'电影', '电视剧', '综艺', '书籍', '音乐', '旅游攻略' 等。
        如果用户没有明确限定类别（例如只说“推荐个好东西”），返回 null。
        """
    )
    platform: Optional[str] = Field(
        None, 
        description="用户提到的特定来源平台，例如：'携程', '马蜂窝', '豆瓣', '知乎'。请提取标准名称或拼音（如 ctrip）。如果没有明确提到平台，必须返回 null。"
    )
    object: Optional[str] = Field(
        None, 
        description="""
        用户想要在标题(title)中进行匹配的【特指关键词】、【名称】或【字符子串】。
        
        【提取规则】：
        1. 提取具体的作品名（如 '肖申克的救赎'）。
        2. 提取特定的限定字符（如 '包含有"王"字的' -> 提取 '王'）。
        3. 提取人名/导演名（如 '周星驰的' -> 提取 '周星驰'）。
        
        【负面约束（Negative Constraints）】：
        - 严禁提取已经在 'category' 字段中出现过的泛化名词（如 '电影', '片子', '书', '攻略'）。
        - 如果用户说 "查一下关于王家卫的电影"，Category提取"电影"，Object提取"王家卫"。
        """
    )
    content_type: Optional[str] = Field(
        None, 
        description="用户想要的内容类型。如果用户明确只看'详情'、'具体内容'、'影评'、'参数'，返回 'child_detail'。如果用户想要'列表'、'概览'、'目录'，返回 'parent_info'。默认返回 null。"
    )
    year: Optional[str] = Field(
        None,
        description="用户提到的具体年份，例如 '2024', '2023'。如果提到'最新'，可以推断为当前年份。没有则返回 null。"
    )

class QueryAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0, # 保持 0 温度以获得最稳定的提取结果
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL
        )
        self.structured_llm = self.llm.with_structured_output(MetadataFilter)
        
        # 【关键】：类别映射字典
        # 将 LLM 提取的自然语言映射为数据库中的标准字段值
        self.CATEGORY_MAPPING = {
            "电影": "movie",
            "影片": "movie",
            "片子": "movie",
            "电视剧": "tv_show",
            "剧集": "tv_show",
            "综艺": "variety",
            "攻略": "guide",
            "文章": "article"
        }

    def generate_expr(self, question: str) -> str:
        print(f"🕵️ Analyzing query: {question}")
        try:
            # 1. 调用 LLM 提取元数据
            prompt_text = QUERY_ANALYZER_PROMPT.format(question=question)
            filter_params: MetadataFilter = self.structured_llm.invoke(prompt_text)
            
            print(f"   📋 Raw Intent: {filter_params.model_dump(exclude_none=True)}")

            # 2. 构建表达式
            expr_parts = []

            # --- A. 处理 Category (新增逻辑) ---
            if filter_params.category:
                # 模糊匹配映射，或者直接使用提取值
                # 这里做一个简单的包含检查，或者直接查字典
                db_category = self.CATEGORY_MAPPING.get(filter_params.category)
                if not db_category:
                    # 如果字典里没有，尝试直接使用提取值，或者记录日志
                    # 这里为了演示，假设直接使用 LLM 提取的词（生产环境建议必须映射）
                    db_category = filter_params.category 
                
                # 假设数据库字段叫 'category'
                expr_parts.append(f"category == '{db_category}'")

            # --- B. 处理 Object (Title 匹配) ---
            if filter_params.object:
                # 清洗引号等脏字符
                clean_obj = filter_params.object.replace("'", "").replace('"', "")
                if clean_obj:
                    expr_parts.append(f"title like '%{clean_obj}%'")

            # --- C. 处理 Platform ---
            if filter_params.platform:
                p = filter_params.platform.lower()
                if "携程" in p: p = "ctrip" # 简单归一化
                expr_parts.append(f"source like '%{p}%'")
            
            # --- D. 处理 Year ---
            if filter_params.year:
                expr_parts.append(f"year == {filter_params.year}") # 假设 year 是 int 或 str

            # 3. 组合
            final_expr = " and ".join(expr_parts)
            
            if final_expr:
                print(f"🎯 Generated SQL/Expr: \"{final_expr}\"")
            else:
                print("   -> No filter, full search.")
            
            return final_expr

        except Exception as e:
            print(f"⚠️ Analysis failed: {e}")
            return ""
        
# 单例模式
query_analyzer = QueryAnalyzer()
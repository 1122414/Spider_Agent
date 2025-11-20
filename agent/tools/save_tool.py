import os
import json
import csv
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from typing import List, Dict, Any, Union

# 引入 registry 以获取缓存数据
from agent.tools.registry import tool_registry

# 确保输出目录存在
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def _get_timestamp_filename(prefix: str, extension: str) -> str:
    """生成带有时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清洗文件名，防止非法字符
    clean_prefix = "".join([c if c.isalnum() else "_" for c in prefix])
    return os.path.join(OUTPUT_DIR, f"{clean_prefix}_{timestamp}.{extension}")

def _resolve_data(data: Union[Dict, List, None]) -> Union[Dict, List, None]:
    """
    【核心修复】数据解析逻辑
    策略：优先使用 Registry 中的缓存数据（完整数据总线）。
    原因：LLM 的上下文窗口有限，它传入的 'data' 参数往往是被截断的（如只包含前2条）。
    只有当缓存为空时，才尝试使用 LLM 传入的 data。
    """
    # 1. 优先检查缓存 (数据总线)
    cached_data = tool_registry.last_execution_result
    
    if cached_data:
        print(f"🔎 [SaveTool] 检测到内存缓存数据 (Type: {type(cached_data)})。")
        print("   -> 为防止数据截断，忽略 LLM 传入的 data 参数，强制使用缓存的完整数据。")
        return cached_data

    # 2. 如果缓存为空（比如直接调用的 save 而没经过爬虫），才使用传入参数
    if data:
        print("🔎 [SaveTool] 缓存为空，使用传入的 data 参数。")
        return data
    
    return None

def save_to_json(data: Dict[str, Any] = None, filename_prefix: str = "crawl_result") -> str:
    """
    保存数据为 JSON 文件
    参数:
        data: (可选) 即使 LLM 传入了此参数，只要上一步有缓存，也会被忽略。
        filename_prefix: 文件名前缀
    """
    # 解析数据 (优先取缓存)
    actual_data = _resolve_data(data)
    
    if not actual_data:
        return "保存失败: 没有数据可保存 (参数为空且无缓存)"

    file_path = _get_timestamp_filename(filename_prefix, "json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(actual_data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 文件已保存: {file_path}")
        return f"成功保存 JSON 到: {file_path}"
    except Exception as e:
        return f"保存 JSON 失败: {str(e)}"

def save_to_csv(data: Union[Dict, List] = None, filename_prefix: str = "crawl_result") -> str:
    """
    保存数据为 CSV 文件
    """
    actual_data = _resolve_data(data)
    
    if not actual_data:
        return "保存失败: 没有数据可保存"

    file_path = _get_timestamp_filename(filename_prefix, "csv")
    
    # 提取列表数据
    items = []
    if isinstance(actual_data, dict):
        # 优先查找 data 或 items 字段
        items = actual_data.get("data") or actual_data.get("items") or actual_data.get("target_content") or []
        # 如果数据本身就是单条字典，且不在 items 里
        if not items and "url" in actual_data:
             items = [actual_data]
    elif isinstance(actual_data, list):
        items = actual_data
        
    if not items:
        return f"保存 CSV 失败: 数据格式不包含列表项 (Keys: {list(actual_data.keys()) if isinstance(actual_data, dict) else 'List'})"

    try:
        # 使用 Pandas 进行智能转换
        df = pd.DataFrame(items)
        
        # 强制将所有非基础类型的列转换为字符串，防止 CSV 写入报错
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
        
        df.to_csv(file_path, index=False, encoding='utf-8-sig') # sig 用于解决 Excel 中文乱码
        print(f"💾 CSV 文件已保存: {file_path} (共 {len(df)} 条)")
        return f"成功保存 CSV 到: {file_path}"
    except Exception as e:
        return f"保存 CSV 失败: {str(e)}"

def save_to_postgres(data: Union[Dict, List] = None, table_name: str = "crawled_data") -> str:
    """
    保存数据到 PostgreSQL 数据库
    """
    actual_data = _resolve_data(data)
    
    if not actual_data:
        return "保存失败: 没有数据可保存"

    # 从环境变量获取数据库连接串
    dsn = os.environ.get("POSTGRES_CONNECTION_STRING")
    if not dsn:
        return "保存失败: 未设置 POSTGRES_CONNECTION_STRING 环境变量"

    # 提取要保存的列表
    items = []
    source_url = ""
    
    if isinstance(actual_data, dict):
        items = actual_data.get("data") or actual_data.get("items") or actual_data.get("target_content") or []
        source_url = actual_data.get("root_url") or actual_data.get("url") or ""
    elif isinstance(actual_data, list):
        items = actual_data

    if not items:
        return "保存数据库失败: 数据为空"

    conn = None
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        # 1. 建表 (如果不存在)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            source_url TEXT,
            crawled_at TIMESTAMP DEFAULT NOW(),
            content JSONB
        );
        """
        cur.execute(create_table_sql)
        
        # 2. 批量插入
        insert_sql = f"INSERT INTO {table_name} (source_url, content) VALUES (%s, %s)"
        
        count = 0
        for item in items:
            cur.execute(insert_sql, (source_url, Json(item)))
            count += 1
            
        conn.commit()
        print(f"💾 已将 {count} 条数据存入数据库表: {table_name}")
        return f"成功将 {count} 条数据存入 PostgreSQL 表 '{table_name}'"

    except Exception as e:
        if conn:
            conn.rollback()
        return f"数据库操作失败: {str(e)}"
    finally:
        if conn:
            conn.close()
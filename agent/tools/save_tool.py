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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_prefix = "".join([c if c.isalnum() else "_" for c in prefix])
    return os.path.join(OUTPUT_DIR, f"{clean_prefix}_{timestamp}.{extension}")

def _extract_items(data: Any) -> List[Dict]:
    """辅助函数：统一从 dict/list 中提取 items 列表"""
    if isinstance(data, dict):
        return data.get("data") or data.get("items") or data.get("target_content") or []
    if isinstance(data, list):
        return data
    return []

def _resolve_data(data: Union[Dict, List, None]) -> Union[Dict, List, None]:
    """
    【核心修复】智能数据解析逻辑
    不再无脑优先缓存，而是判断传入数据的有效性。
    """
    cached_data = tool_registry.last_execution_result
    
    # 1. 如果没传 data，必须用缓存
    if not data:
        if cached_data:
             print("🔎 [SaveTool] 未传入 data，自动使用内存缓存。")
             return cached_data
        return None

    # 2. 如果传了 data，且有缓存，进行智能比对
    if cached_data:
        # 如果缓存是错误信息（字符串），直接忽略缓存，使用 data
        if isinstance(cached_data, str):
             return data

        items_passed = _extract_items(data)
        items_cached = _extract_items(cached_data)
        
        # 【判定逻辑】什么时候该用缓存覆盖传入的数据？
        # 只有当：传入数据量极少（<5） AND 缓存数据量大（>5） AND 看起来像是缓存的子集（截断）
        if items_passed and items_cached:
            if len(items_passed) <= 5 and len(items_cached) > 5:
                # 进一步检查内容是否相似 (比对第一条数据)
                try:
                    # 简单比对第一条数据的名称或链接是否一致
                    first_passed = items_passed[0].get("链接") or items_passed[0].get("url") or items_passed[0].get("link")
                    first_cached = items_cached[0].get("链接") or items_cached[0].get("url") or items_cached[0].get("link")
                    
                    if first_passed and first_passed == first_cached:
                        print(f"🔎 [SaveTool] 检测到传入数据 ({len(items_passed)}条) 可能是缓存 ({len(items_cached)}条) 的截断版本，自动切换为完整缓存。")
                        return cached_data
                except:
                    pass

        # 其他情况（例如传入数据有 children 而缓存没有，或者数据完全不同），尊重传入的 data
        print(f"🔎 [SaveTool] 尊重传入的 data 参数 (Items: {len(items_passed)})，忽略缓存冲突。")
        return data

    # 3. 无缓存，直接用 data
    return data

def save_to_json(data: Dict[str, Any] = None, filename_prefix: str = "crawl_result") -> str:
    actual_data = _resolve_data(data)
    
    if not actual_data:
        return "保存失败: 没有数据可保存"

    file_path = _get_timestamp_filename(filename_prefix, "json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(actual_data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 文件已保存: {file_path}")
        return f"成功保存 JSON 到: {file_path}"
    except Exception as e:
        return f"保存 JSON 失败: {str(e)}"

def save_to_csv(data: Union[Dict, List] = None, filename_prefix: str = "crawl_result") -> str:
    actual_data = _resolve_data(data)
    
    if not actual_data:
        return "保存失败: 没有数据可保存"
    
    if isinstance(actual_data, str):
        return f"保存失败: 数据格式错误 (String): {actual_data[:50]}..."

    file_path = _get_timestamp_filename(filename_prefix, "csv")
    items = _extract_items(actual_data)
    
    # 如果 items 为空但 actual_data 本身是单条数据
    if not items and isinstance(actual_data, dict) and "url" in actual_data:
        items = [actual_data]
        
    if not items:
        return f"保存 CSV 失败: 数据中未找到列表项。"

    try:
        df = pd.DataFrame(items)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
        
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"💾 CSV 文件已保存: {file_path} (共 {len(df)} 条)")
        return f"成功保存 CSV 到: {file_path}"
    except Exception as e:
        return f"保存 CSV 失败: {str(e)}"

def save_to_postgres(data: Union[Dict, List] = None, table_name: str = "crawled_data") -> str:
    actual_data = _resolve_data(data)
    if not actual_data or isinstance(actual_data, str):
        return "保存失败: 没有有效数据"

    dsn = os.environ.get("POSTGRES_CONNECTION_STRING")
    if not dsn: return "保存失败: 未设置 POSTGRES_CONNECTION_STRING"

    items = _extract_items(actual_data)
    source_url = actual_data.get("root_url", "") if isinstance(actual_data, dict) else ""

    if not items: return "保存数据库失败: 数据为空"

    conn = None
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                source_url TEXT,
                crawled_at TIMESTAMP DEFAULT NOW(),
                content JSONB
            );
        """)
        for item in items:
            cur.execute(f"INSERT INTO {table_name} (source_url, content) VALUES (%s, %s)", (source_url, Json(item)))
        conn.commit()
        print(f"💾 已将 {len(items)} 条数据存入数据库: {table_name}")
        return f"成功将 {len(items)} 条数据存入 DB"
    except Exception as e:
        if conn: conn.rollback()
        return f"数据库操作失败: {str(e)}"
    finally:
        if conn: conn.close()
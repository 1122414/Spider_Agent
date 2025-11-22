import random
from pymilvus import connections, Collection, utility

def inspect_milvus_advanced():
    print("🔌 正在连接 Milvus (localhost:19530)...")
    try:
        connections.connect(alias="default", host="localhost", port="19530")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    target_collection = "spider_knowledge_base"
    if not utility.has_collection(target_collection):
        print(f"⚠️ 集合 '{target_collection}' 不存在。")
        return

    # 加载集合
    collection = Collection(target_collection)
    print("🌊 正在刷新数据 (Flush)...")
    collection.flush() # 强制刷盘，确保计数准确
    collection.load()
    
    # 1. 总量核对
    total_count = collection.num_entities
    print(f"\n📊 【数据总量】: {total_count} 条 Document")
    
    if total_count == 0:
        print("⚠️ 集合为空，请先执行爬取和入库任务。")
        return

    # 2. 统计电影数量 (通过查询 title 字段)
    print("\n🧮 正在进行去重统计 (Group By Title)...")
    # 获取所有数据的 title 字段 (注意：如果数据量极大，这里需要分批，但演示场景够用了)
    try:
        # 限制取前 1000 条做统计预览
        results = collection.query(
            expr="pk >= 0", 
            limit=1000,  
            output_fields=["title", "type"]
        )
        
        titles = set()
        parent_count = 0
        child_count = 0
        
        for res in results:
            t = res.get("title", "Unknown")
            titles.add(t)
            if res.get("type") == "parent_info":
                parent_count += 1
            elif res.get("type") == "child_detail":
                child_count += 1
                
        print(f"🎬 【电影总数】: {len(titles)} 部 (基于前 {len(results)} 条数据统计)")
        print(f"   - 基础信息 (Parent): {parent_count} 条")
        print(f"   - 剧情详情 (Child) : {child_count} 条")
        
        print("\n📋 电影清单 (前 10 部):")
        for i, t in enumerate(list(titles)[:10]):
            print(f"   {i+1}. {t}")
            
        if len(titles) > 10:
            print(f"   ... 以及其他 {len(titles)-10} 部")

    except Exception as e:
        print(f"统计失败: {e}")

    # 3. 随机抽样展示
    print(f"\n🎲 【随机抽样展示 2 条】:")
    # 随机取一个偏移量
    random_offset = random.randint(0, max(0, total_count - 1))
    sample_results = collection.query(
        expr="pk >= 0",
        limit=2,
        offset=random_offset, # 跳过前面的数据，看后面的
        output_fields=["title", "text", "source"]
    )
    
    for res in sample_results:
        print("-" * 50)
        print(f"Title : {res.get('title')}")
        print(f"Source: {res.get('source')}")
        print(f"Text  : {res.get('text')[:100]}...") # 只显示前100字

if __name__ == "__main__":
    inspect_milvus_advanced()
# ```

# ### 预期结果
# 运行这个新脚本：
# ```bash
# python utils/check_milvus.py
# ```

# 你应该能看到类似这样的输出：
# ```text
# 📊 【数据总量】: 200 条 Document (假设你爬了100部电影 x 2)
# 🎬 【电影总数】: 100 部
#    - 基础信息 (Parent): 100 条
#    - 剧情详情 (Child) : 100 条
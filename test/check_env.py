import sys
import importlib

def check_import(module_name, class_name=None):
    try:
        module = importlib.import_module(module_name)
        print(f"✅ 成功导入模块: {module_name}")
        if class_name:
            if hasattr(module, class_name):
                print(f"   -> 成功找到类: {class_name}")
            else:
                print(f"   ❌ 模块中未找到类: {class_name}")
                # 尝试打印模块路径，方便排查
                print(f"      路径: {module.__file__}")
    except ImportError as e:
        print(f"❌ 导入失败: {module_name}")
        print(f"   错误信息: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

print(f"当前 Python 解释器路径: {sys.executable}")
print("-" * 50)

# 1. 检查 langchain 主包
check_import("langchain")

# 2. 检查检索器模块
check_import("langchain_community.retrievers", "ContextualCompressionRetriever")

# 3. 检查文档压缩器模块
check_import("langchain_community.document_compressors", "CrossEncoderReranker")

# 4. 检查社区包 HuggingFace
check_import("langchain_community.cross_encoders", "HuggingFaceCrossEncoder")

print("-" * 50)
# ```

# **运行方法：**
# 在终端（Terminal）中确保激活了 `spider_agent` 环境，然后运行：
# ```bash
# python check_env.py
# ```

# * **如果全绿（✅）**：说明环境完全没问题。请执行 **第三步** 修复编辑器显示。
# * **如果有红（❌）**：说明包确实没装好或者装错位置了。请执行 **第二步**。

# ---

# #### 第二步：强制重装 (如果脚本报错)

# 如果 `check_env.py` 报错说找不到模块，说明 `pip` 可能把包安装到了其他地方，或者之前的安装过程有缓存干扰。

# 请在终端运行以下**强制重装**命令：

# ```bash
# # 1. 先卸载可能冲突的旧包
# pip uninstall -y langchain langchain-community langchain-core

# # 2. 重新安装指定最新版本
# pip install langchain langchain-community langchain-core sentence-transformers --no-cache-dir
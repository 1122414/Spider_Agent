import requests
import json
import time
import os
import sys

# ================= 配置区域 =================
VLLM_HOST = "http://172.23.216.104:5005" 
# 注意：你的server.py中定义的model默认值是 "Qwen3-Embedding-8B"
# 客户端请求时最好保持一致，虽然你的简易服务端目前没有校验这个字段
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"    
API_KEY = "EMPTY"                         
# ===========================================

def print_separator(title):
    print("\n" + "=" * 50)
    print(f"🕵️  {title}")
    print("=" * 50)

def check_proxy():
    print_separator("1. 环境检查")
    # 强制清除代理，防止干扰内网访问
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    print("✅ 已强制清除进程内代理设置 (http_proxy/https_proxy)，确保直连局域网 IP")

def test_direct_generation():
    """
    因为服务端是极简版，没有 /v1/models 接口，所以我们直接进行生成测试。
    这也是唯一的'健康检查'方式。
    """
    print_separator("2. 向量生成测试 (核心)")
    
    # 你的 server.py 定义的是 @app.post("/v1/embeddings") <- 注意复数 s
    url = f"{VLLM_HOST}/v1/embeddings"
    
    payload = {
        "model": MODEL_NAME,
        "input": "Sanity check for embedding generation."
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    print(f"🚀 发送请求: POST {url}")
    print(f"📦 Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        session = requests.Session()
        session.trust_env = False # 再次确保忽略系统代理
        
        start_t = time.time()
        # 设置短一点的连接超时(connect timeout)，长一点的读取超时(read timeout)
        # 因为加载模型可能需要时间
        resp = session.post(url, json=payload, headers=headers, timeout=(5, 60))
        cost = time.time() - start_t
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                # 兼容你的 server.py 返回结构
                if 'data' in data and len(data['data']) > 0:
                    vec = data['data'][0]['embedding']
                    print(f"\n✅ [成功] 向量生成成功！")
                    print(f"   ⏱️ 耗时: {cost:.2f}s")
                    print(f"   📏 维度: {len(vec)}")
                    print(f"   👀 前5位: {vec[:5]}")
                    return True
                else:
                    print(f"\n❌ [失败] 响应格式不符合预期: {data}")
            except Exception as parse_err:
                print(f"\n❌ [失败] 解析 JSON 响应出错: {parse_err}")
                print(f"   原始响应: {resp.text}")
        
        elif resp.status_code == 404:
            print(f"\n❌ [404 Not Found] 路径错误")
            print(f"   你请求的地址是: {url}")
            print(f"   请检查 server.py 中 @app.post(...) 里的路径是否也是 /v1/embeddings")
            
        elif resp.status_code == 405:
            print(f"\n❌ [405 Method Not Allowed]")
            print(f"   这通常意味着你用 GET 请求了一个只允许 POST 的接口，或者反之。")
            print(f"   当前代码使用的是: POST")
            
        elif resp.status_code == 500:
            print(f"\n❌ [500 Internal Server Error] 服务端报错")
            print(f"   请查看服务端控制台的 Python 报错日志 (可能是显存不足 OOM)")
            print(f"   响应内容: {resp.text}")
            
        else:
            print(f"\n❌ [失败] 状态码: {resp.status_code}")
            print(f"   响应: {resp.text}")

    except requests.exceptions.ConnectionError:
        print(f"\n❌ [连接拒绝] 无法连接到 {VLLM_HOST}")
        print(f"   1. 请确认 server.py 正在运行且监听 0.0.0.0:5005")
        print(f"   2. 请确认防火墙允许 5005 端口")
        print(f"   3. 尝试在服务器本机运行 `curl http://localhost:5005/v1/embeddings` 测试")
        
    except Exception as e:
        print(f"\n❌ [未知错误] {e}")

if __name__ == "__main__":
    check_proxy()
    test_direct_generation()
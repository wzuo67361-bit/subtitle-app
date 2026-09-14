import os
import google.generativeai as genai

# 配置 API Key
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=api_key)

# 完整的可用模型列表
AVAILABLE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-pro"
]

def generate_content(prompt, model_name="gemini-3.5-flash"):
    """
    使用指定的模型生成内容
    """
    if model_name not in AVAILABLE_MODELS:
        print(f"警告: {model_name} 不在推荐的模型列表中。")
        
    try:
        # 如果需要通过代理请求，可以在环境变量中提前设置 HTTP_PROXY / HTTPS_PROXY
        # os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
        # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"请求失败: {str(e)}"

if __name__ == "__main__":
    # 测试调用
    test_prompt = "你好，请自我介绍一下。"
    selected_model = "gemini-3.8-flash"  # 在这里切换你想测试的模型
    
    print(f"正在使用 {selected_model} 生成回复...\n")
    result = generate_content(test_prompt, selected_model)
    print(result)

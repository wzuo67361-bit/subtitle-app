import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
import requests
import tempfile
import os
import math
import re
import time

# --- 页面配置 ---
st.set_page_config(page_title="音视频字幕生成与翻译", page_icon="🎬", layout="wide")
st.title("🎬 音视频字幕生成与翻译 Web 应用")
st.markdown("支持高精度语音识别、大模型角色语气推断、多语言翻译及双语对照。")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# ==========================================
# 1. 平台与模型选择 (联动菜单)
# ==========================================
st.sidebar.subheader("1. API 设置 (翻译大脑)")

platforms = [
    "Google Gemini",
    "DeepSeek",
    "Kimi (月之暗面)",
    "阿里通义千问 (Qwen)",
    "OpenAI 官方",
    "自定义 (第三方代理/中转)"
]

selected_platform = st.sidebar.selectbox("① 选择大模型平台", platforms)

base_url = ""
model_name = ""
api_key = ""

if selected_platform == "Google Gemini":
    st.sidebar.info(
        "**【Google Gemini 模式】**\n"
        "- 底层已开启 **全协议自适应握手**。\n"
        "- 完美兼容 `AQ.` 新型凭证以及 `AIzaSy` 传统密钥。\n"
        "- 官方提供丰厚的免费请求额度。"
    )
    gemini_models = [
        "gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "gemini-1.0-pro", 
        "gemini-pro"
    ]
    model_name = st.sidebar.selectbox("② 选择具体模型", gemini_models)
    base_url = "native_gemini"
    
elif selected_platform == "DeepSeek":
    st.sidebar.info("**【DeepSeek 模式】** 性价比极高，适合长文本高质量翻译。")
    deepseek_models = ["deepseek-chat", "deepseek-coder"]
    model_name = st.sidebar.selectbox("② 选择具体模型", deepseek_models)
    base_url = "https://api.deepseek.com/v1"
    
elif selected_platform == "Kimi (月之暗面)":
    st.sidebar.info("**【Kimi 模式】** 国内顶尖上下文理解与语气还原模型。")
    kimi_models = ["moonshot-v1-8k", "moonshot-v1-32k"]
    model_name = st.sidebar.selectbox("② 选择具体模型", kimi_models)
    base_url = "https://api.moonshot.cn/v1"
    
elif selected_platform == "阿里通义千问 (Qwen)":
    st.sidebar.info("**【通义千问 模式】** 阿里大厂模型，稳定高速。")
    qwen_models = ["qwen-plus", "qwen-max", "qwen-turbo"]
    model_name = st.sidebar.selectbox("② 选择具体模型", qwen_models)
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
elif selected_platform == "OpenAI 官方":
    st.sidebar.info("**【OpenAI 模式】** 行业基准模型。")
    openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    model_name = st.sidebar.selectbox("② 选择具体模型", openai_models)
    base_url = "https://api.openai.com/v1"
    
else:
    st.sidebar.info("**【自定义模式】** 请填入第三方中转/代理平台的标准地址与模型名。")
    base_url = st.sidebar.text_input("② 输入 API 网址 (Base URL)", placeholder="例如: https://api.siliconflow.cn/v1")
    model_name = st.sidebar.text_input("③ 输入模型名称", placeholder="例如: Qwen/Qwen2.5-7B-Instruct")

api_key = st.sidebar.text_input("最后：输入你的 API Key (支持 AQ... / sk-...)", type="password")

# ==========================================
# 2. 语言与字幕选项
# ==========================================
st.sidebar.subheader("2. 字幕设置")
source_lang = st.sidebar.selectbox("视频源语言", ["ja (日语)", "auto (自动识别)", "en (英语)", "zh (中文)"], index=0)

target_option = st.sidebar.selectbox(
    "目标字幕选项",
    [
        "仅生成日文原字幕 (SRT)",
        "翻译为简体中文 (SRT)",
        "翻译为英文 (SRT)",
        "生成【日/中】双语对照字幕 (SRT)",
        "生成【日/英】双语对照字幕 (SRT)"
    ]
)

# ==========================================
# 3. 专业词汇校正
# ==========================================
st.sidebar.subheader("3. 专业词汇/专有名词校正")
glossary = st.sidebar.text_area(
    "输入翻译对照（如：人名、术语），每行一个",
    placeholder="例如：\n山田太郎 -> Yamada Taro\n术语A -> Term A",
    height=100
)

# --- 核心处理函数 ---

def format_timestamp(seconds: float):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

def call_gemini_native(prompt_text, user_content, raw_key, model):
    """Google Gemini 自适应握手协议：同时支持 x-goog-api-key, Bearer Token 与 Query 传参"""
    clean_model = model.replace("models/", "").strip()
    key = raw_key.strip()
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
    full_instruction = f"{prompt_text}\n\n【待处理 SRT 字幕如下】：\n{user_content}"
    
    payload = {
        "contents": [{"parts": [{"text": full_instruction}]}],
        "generationConfig": {"temperature": 0.3}
    }
    
    # 策略 1：使用 Google 官方推荐的 x-goog-api-key Header（专门接纳新型 AQ. 凭证）
    headers_strategy_1 = {
        "Content-Type": "application/json",
        "x-goog-api-key": key
    }
    
    try:
        res = requests.post(endpoint, headers=headers_strategy_1, json=payload, timeout=90)
        
        # 策略 2：如果策略 1 报 401，尝试 Authorization: Bearer OAuth 认证头
        if res.status_code == 401:
            headers_strategy_2 = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
            res = requests.post(endpoint, headers=headers_strategy_2, json=payload, timeout=90)
            
        # 策略 3：如果依然 401，尝试老版传统的 URL 问号传参
        if res.status_code == 401:
            url_strategy_3 = f"{endpoint}?key={key}"
            res = requests.post(url_strategy_3, headers={"Content-Type": "application/json"}, json=payload, timeout=90)

        # 结果解析
        if res.status_code == 200:
            data = res.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            result = re.sub(r'^```(?:srt|text)?\n', '', result)
            return re.sub(r'\n```$', '', result)
        else:
            return f"翻译出错: Google 服务器响应异常 (代码 {res.status_code}) - {res.text}"
            
    except Exception as e:
        return f"翻译出错 (网络异常): {str(e)}"

def call_openai_compatible(prompt_text, user_content, key, url, model):
    """标准 OpenAI 兼容通道"""
    try:
        client = OpenAI(api_key=key.strip(), base_url=url.strip())
        response = client.chat.completions.create(
            model=model.strip(),
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        result = re.sub(r'^```(?:srt|text)?\n', '', result)
        return re.sub(r'\n```$', '', result)
    except Exception as e:
        return f"翻译出错: {str(e)}"

# --- 主界面执行逻辑 ---

st.write("### 📤 第一步：上传音视频文件")
uploaded_file = st.file_uploader("支持 MP4, MP3, WAV, M4A 等主流音视频格式", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("🚀 开始生成与翻译", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ 请先上传音视频文件！")
        st.stop()
    
    if "翻译" in target_option or "双语" in target_option:
        if not api_key or not model_name:
            st.warning("⚠️ 请在左侧侧边栏填入 API Key！")
            st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        st.write("### ⏳ 第二步：处理进度")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. Faster-Whisper 音频提取
        status_text.info("🎧 正在使用 faster-whisper 提取原字幕 (请稍候)...")
        model = load_whisper_model()
        lang_code = source_lang.split(" ")[0]
        lang_param = None if lang_code == "auto" else lang_code
        
        segments, info = model.transcribe(tmp_file_path, language=lang_param, beam_size=5)
        
        original_srt_lines = []
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            original_srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
            
        original_srt_text = "\n".join(original_srt_lines)
        progress_bar.progress(50)

        # 2. AI 智能翻译与语气校正
        final_srt_text = original_srt_text
        
        if "翻译" in target_option or "双语" in target_option:
            status_text.info(f"🧠 正在调用大模型 ({model_name}) 深入解析对话与语气...")
            
            system_prompt = f"""你是一个顶级的影视字幕翻译专家。目标任务：{target_option}。
专业词汇校对对照表：\n{glossary}

【核心翻译原则 - 智能角色与语气还原】：
1. 必须通盘理解日文上下文，根据自称（俺、僕、私、あたし等）、句尾终助词（わ、ぜ、ぞ、かしら等）以及敬语/简体的差异，精准推断说话人的性别与身份关系。
2. 翻译出的译文必须符合该角色的性格与语气！男性台词坚决展现男人口吻，女性台词体现女性口吻，坚决杜绝生硬机翻。
3. 严格保留原有的 SRT 序号和时间轴格式（如 1 \\n 00:00:01,000 --> 00:00:04,000）。
4. 若选择双语，第一行为原文，第二行为译文。
5. 绝对不要输出任何 Markdown 标记（如 ```srt），直接输出纯文本。"""

            chunk_size = 35
            translated_srt_pieces = []
            total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
            
            for i in range(total_chunks):
                chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                chunk_text = "\n".join(chunk_lines)
                status_text.info(f"🧠 正在翻译第 {i+1}/{total_chunks} 组字幕 (角色语气分析中)...")
                
                # 路由判断
                if base_url == "native_gemini":
                    translated_chunk = call_gemini_native(system_prompt, chunk_text, api_key, model_name)
                    time.sleep(2) # 避免触发频率限制
                else:
                    translated_chunk = call_openai_compatible(system_prompt, chunk_text, api_key, base_url, model_name)
                
                if "翻译出错" in translated_chunk:
                    st.error(translated_chunk)
                    st.stop()
                    
                translated_srt_pieces.append(translated_chunk)
                current_progress = 50 + int(50 * ((i + 1) / total_chunks))
                progress_bar.progress(current_progress)
                
            final_srt_text = "\n\n".join(translated_srt_pieces)
        else:
            progress_bar.progress(100)

        status_text.success("✅ 全部处理完成！请在下方预览并下载字幕。")

        # 3. 预览与下载
        st.write("---")
        st.write("### 👀 第三步：字幕预览与下载")
        st.text_area("字幕内容确认区（可直接在此处二次编辑）：", final_srt_text, height=400)
        
        st.download_button(
            label="⬇️ 一键下载 .srt 字幕文件",
            data=final_srt_text,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ 处理过程中发生异常: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

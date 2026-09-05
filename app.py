import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
import requests
import tempfile
import os
import math
import re
import json

# --- 页面配置 ---
st.set_page_config(page_title="音视频字幕生成与翻译", page_icon="🎬", layout="wide")
st.title("🎬 音视频字幕生成与翻译 Web 应用")
st.markdown("支持高精度语音识别、大模型角色语气推断、多语言翻译及双语对照。")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# 1. API 配置
st.sidebar.subheader("1. API 设置 (翻译大脑)")

platform_options = [
    "Google Gemini (官方推荐)",
    "DeepSeek (高性价比)",
    "Kimi (月之暗面)",
    "阿里通义千问 (Qwen)",
    "OpenAI 官方",
    "自定义 (兼容 OpenAI 格式平台)"
]

selected_provider = st.sidebar.selectbox("选择大模型平台", platform_options)

# 针对各平台的参数预设
if selected_provider == "Google Gemini (官方推荐)":
    st.sidebar.caption("⚡ 采用 Google 原生 v1beta 专线，解决 404 路由问题")
    model_name = st.sidebar.text_input("模型名称", value="gemini-1.5-flash")
    base_url = "" # Gemini 走专属原生通道
elif selected_provider == "DeepSeek (高性价比)":
    base_url = st.sidebar.text_input("API 网址", value="https://api.deepseek.com/v1")
    model_name = st.sidebar.text_input("模型名称", value="deepseek-chat")
elif selected_provider == "Kimi (月之暗面)":
    base_url = st.sidebar.text_input("API 网址", value="https://api.moonshot.cn/v1")
    model_name = st.sidebar.text_input("模型名称", value="moonshot-v1-8k")
elif selected_provider == "阿里通义千问 (Qwen)":
    base_url = st.sidebar.text_input("API 网址", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = st.sidebar.text_input("模型名称", value="qwen-plus")
elif selected_provider == "OpenAI 官方":
    base_url = st.sidebar.text_input("API 网址", value="https://api.openai.com/v1")
    model_name = st.sidebar.text_input("模型名称", value="gpt-4o-mini")
else:
    base_url = st.sidebar.text_input("自定义 API 网址 (Base URL)", placeholder="https://api.example.com/v1")
    model_name = st.sidebar.text_input("自定义模型名称", placeholder="例如: gpt-3.5-turbo")

api_key = st.sidebar.text_input("输入对应的 API Key", type="password", placeholder="填入你的密钥...")

# 2. 语言与字幕选项
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

# 3. 专业词汇校正
st.sidebar.subheader("3. 专业词汇/专有名词校正")
glossary = st.sidebar.text_area(
    "输入翻译对照（如：人名、术语），每行一个",
    placeholder="例如：\n山田太郎 -> Yamada Taro\n术语A -> Term A",
    height=100
)

# --- 核心处理函数 ---

def format_timestamp(seconds: float):
    """时间戳转 SRT 格式 (HH:MM:SS,mmm)"""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

@st.cache_resource
def load_whisper_model():
    """加载 faster-whisper 引擎"""
    return WhisperModel("base", device="cpu", compute_type="int8")

def call_gemini_native(prompt_text, user_content, key, model):
    """Google Gemini 原生 v1beta 直连通道，彻底根绝 404/v1main 错误"""
    clean_model = model.replace("models/", "").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key.strip()}"
    
    full_instruction = f"{prompt_text}\n\n【待处理 SRT 字幕如下】：\n{user_content}"
    
    payload = {
        "contents": [
            {
                "parts": [{"text": full_instruction}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        if response.status_code != 200:
            return f"翻译出错 (Google 返回错误): {response.text}"
        
        data = response.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        result = re.sub(r'^```(?:srt|text)?\n', '', result)
        result = re.sub(r'\n```$', '', result)
        return result
    except Exception as e:
        return f"翻译出错 (Gemini 请求异常): {str(e)}"

def call_openai_compatible(prompt_text, user_content, key, url, model):
    """OpenAI 兼容协议通道 (适用 DeepSeek, Kimi, 阿里, OpenAI)"""
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
        result = re.sub(r'\n```$', '', result)
        return result
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
        if not api_key:
            st.warning("⚠️ 请在左侧侧边栏填入对应的 API Key！")
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

            chunk_size = 35 # 适中分块，保障翻译上下文同时杜绝截断
            translated_srt_pieces = []
            total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
            
            for i in range(total_chunks):
                chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                chunk_text = "\n".join(chunk_lines)
                status_text.info(f"🧠 正在翻译第 {i+1}/{total_chunks} 组字幕 (角色语气分析中)...")
                
                # 路由判断：Google Gemini 走原生专线，其他平台走 OpenAI 协议
                if selected_provider == "Google Gemini (官方推荐)":
                    translated_chunk = call_gemini_native(system_prompt, chunk_text, api_key, model_name)
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

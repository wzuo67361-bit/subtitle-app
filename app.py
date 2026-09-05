import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
from deep_translator import GoogleTranslator
import tempfile
import os
import math
import re

# --- 页面配置 ---
st.set_page_config(page_title="音视频字幕生成与翻译", page_icon="🎬", layout="wide")
st.title("🎬 音视频字幕生成与翻译 Web 应用")
st.markdown("支持提取原字幕、多语言翻译、双语对照。")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# 1. 翻译引擎选择 (核心升级)
st.sidebar.subheader("1. 选择翻译引擎")
engine_choice = st.sidebar.radio(
    "请选择你要使用的翻译方式：",
    ["🟢 免费基础版 (无需密钥，直接可用)", "🚀 AI 高级版 (需填密钥，精准语气)"]
)

api_key = ""
base_url = ""
model_name = ""

if engine_choice == "🚀 AI 高级版 (需填密钥，精准语气)":
    st.sidebar.info("高级版支持识别角色语气，但需要填写 API 密钥。")
    api_presets = {
        "Kimi (月之暗面)": {"url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
        "阿里通义千问": {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
        "DeepSeek": {"url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
        "自定义": {"url": "", "model": ""}
    }
    selected_provider = st.sidebar.selectbox("选择大模型平台", list(api_presets.keys()))
    
    if selected_provider == "自定义":
        base_url = st.sidebar.text_input("API 网址", placeholder="https://...")
        model_name = st.sidebar.text_input("模型名称", placeholder="例如: gpt-3.5-turbo")
    else:
        base_url = st.sidebar.text_input("API 网址", value=api_presets[selected_provider]["url"])
        model_name = st.sidebar.text_input("模型名称", value=api_presets[selected_provider]["model"])
        
    api_key = st.sidebar.text_input("输入 API Key (sk-...)", type="password")
else:
    st.sidebar.success("✅ 当前为免密钥模式，直接上传文件即可运行！")

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

def translate_with_ai(srt_chunk, target_opt, client, model):
    """AI 高级翻译"""
    system_prompt = f"""你是一个专业的字幕翻译专家。目标：{target_opt}。
请根据原文语气（如男女自称、口语）精准翻译。严格保留SRT时间轴格式，不要输出markdown。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": srt_chunk}],
            temperature=0.3
        )
        res = response.choices[0].message.content.strip()
        res = re.sub(r'^```(?:srt|text)?\n', '', res)
        return re.sub(r'\n```$', '', res)
    except Exception as e:
        return f"翻译出错: {str(e)}"

def translate_with_free_engine(original_lines, target_opt):
    """免费基础翻译 (无密钥)"""
    target_lang = 'zh-CN' if '中' in target_opt else 'en'
    translator = GoogleTranslator(source='auto', target=target_lang)
    
    translated_blocks = []
    for block in original_lines:
        parts = block.strip().split('\n')
        if len(parts) >= 3:
            idx = parts[0]
            timing = parts[1]
            text = "\n".join(parts[2:])
            
            try:
                trans_text = translator.translate(text)
            except:
                trans_text = text # 翻译失败则保留原文
                
            if "双语" in target_opt:
                final_text = f"{text}\n{trans_text}"
            else:
                final_text = trans_text
                
            translated_blocks.append(f"{idx}\n{timing}\n{final_text}\n")
            
    return "\n".join(translated_blocks)

# --- 主界面逻辑 ---

st.write("### 📤 第一步：上传文件")
uploaded_file = st.file_uploader("支持 MP4, MP3, WAV, M4A 等格式", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("🚀 开始生成与翻译", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ 请先上传音视频文件！")
        st.stop()
    
    if "翻译" in target_option or "双语" in target_option:
        if engine_choice == "🚀 AI 高级版 (需填密钥，精准语气)" and not api_key:
            st.warning("⚠️ 高级版需要填写 API Key！或者请切换到【免费基础版】。")
            st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        st.write("### ⏳ 第二步：处理进度")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. 提取原字幕
        status_text.info("🎧 正在提取语音并生成原字幕 (这可能需要几分钟，请耐心等待)...")
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
            
        progress_bar.progress(50)

        # 2. 翻译字幕
        final_srt_text = "\n".join(original_srt_lines)
        
        if "翻译" in target_option or "双语" in target_option:
            if engine_choice == "🟢 免费基础版 (无需密钥，直接可用)":
                status_text.info("🌐 正在使用免费引擎进行翻译...")
                final_srt_text = translate_with_free_engine(original_srt_lines, target_option)
                progress_bar.progress(100)
            else:
                status_text.info(f"🧠 正在调用 AI ({model_name}) 进行高级翻译...")
                client = OpenAI(api_key=api_key, base_url=base_url)
                chunk_size = 40
                translated_srt_pieces = []
                total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
                
                for i in range(total_chunks):
                    chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                    chunk_text = "\n".join(chunk_lines)
                    status_text.info(f"🧠 正在翻译第 {i+1}/{total_chunks} 部分...")
                    translated_chunk = translate_with_ai(chunk_text, target_option, client, model_name)
                    
                    if "翻译出错" in translated_chunk:
                        st.error(translated_chunk)
                        st.stop()
                        
                    translated_srt_pieces.append(translated_chunk)
                    current_progress = 50 + int(50 * ((i + 1) / total_chunks))
                    progress_bar.progress(current_progress)
                    
                final_srt_text = "\n\n".join(translated_srt_pieces)
        else:
            progress_bar.progress(100)

        status_text.success("✅ 处理完成！请在下方预览并下载字幕。")

        # 3. 预览与下载
        st.write("---")
        st.write("### 👀 第三步：字幕预览与下载")
        st.text_area("你可以直接在这里检查、修改生成的字幕内容：", final_srt_text, height=400)
        
        st.download_button(
            label="⬇️ 确认无误，一键下载 .srt 字幕文件",
            data=final_srt_text,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ 处理过程中发生错误: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

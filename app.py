import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI
import tempfile
import os
import math
import re

# --- 页面配置 ---
st.set_page_config(page_title="音视频字幕生成与翻译", page_icon="🎬", layout="wide")
st.title("🎬 音视频字幕生成与翻译 Web 应用")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# 1. API 配置
st.sidebar.subheader("API 设置")
api_provider = st.sidebar.radio("选择大模型提供商", ["OpenAI", "DeepSeek"])
api_key = st.sidebar.text_input("输入 API Key", type="password", placeholder="sk-...")
if api_provider == "DeepSeek":
    base_url = "https://api.deepseek.com/v1"
    model_name = "deepseek-chat"
else:
    base_url = "https://api.openai.com/v1"
    model_name = "gpt-4o-mini" # 默认使用高性价比模型

# 2. 语言与字幕选项
st.sidebar.subheader("字幕设置")
source_lang = st.sidebar.selectbox(
    "源语言", 
    ["ja (日语)", "auto (自动识别)", "en (英语)", "zh (中文)"], 
    index=0
)

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
st.sidebar.subheader("专业词汇/专有名词校正")
glossary = st.sidebar.text_area(
    "输入翻译对照（如：人名、术语），每行一个",
    placeholder="例如：\n山田太郎 -> Yamada Taro\n术语A -> Term A",
    height=100
)

# --- 核心处理函数 ---

def format_timestamp(seconds: float):
    """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

@st.cache_resource
def load_whisper_model():
    """加载 faster-whisper 模型 (使用 base 模型以适应免费云端资源)"""
    # 在 CPU 环境下使用 int8，如果在有 GPU 的环境可改为 device="cuda", compute_type="float16"
    return WhisperModel("base", device="cpu", compute_type="int8")

def translate_srt_chunk(srt_chunk, target_opt, gloss, client, model):
    """调用 LLM 翻译 SRT 文本块"""
    system_prompt = f"""你是一个专业的字幕翻译专家。请严格按照要求翻译以下 SRT 字幕。
目标要求：{target_opt}
专业词汇表：\n{gloss}

【严格规则】：
1. 必须严格保留原有的 SRT 序号和时间轴（如 1 \n 00:00:01,000 --> 00:00:04,000）。
2. 如果要求双语，请将原文放在第一行，译文放在第二行。
3. 不要输出任何 Markdown 标记（如 ```srt），直接输出纯文本。
4. 不要合并或删除任何时间轴。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": srt_chunk}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        # 清理可能出现的 markdown 标记
        result = re.sub(r'^```(?:srt|text)?\n', '', result)
        result = re.sub(r'\n```$', '', result)
        return result
    except Exception as e:
        return f"翻译出错: {str(e)}"

# --- 主界面逻辑 ---

uploaded_file = st.file_uploader("上传音视频文件 (支持 MP4, MP3, WAV, M4A)", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("🚀 开始处理", type="primary"):
    if not uploaded_file:
        st.warning("请先上传文件！")
        st.stop()
    
    if "翻译" in target_option or "双语" in target_option:
        if not api_key:
            st.warning("请在左侧输入 API Key 以进行翻译！")
            st.stop()

    # 1. 保存上传的文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 2. 语音识别 (Whisper)
        status_text.info("⏳ 正在提取语音并生成原字幕 (这可能需要几分钟)...")
        model = load_whisper_model()
        
        lang_code = source_lang.split(" ")[0]
        lang_param = None if lang_code == "auto" else lang_code
        
        segments, info = model.transcribe(tmp_file_path, language=lang_param, beam_size=5)
        
        original_srt_lines = []
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            
            srt_block = f"{i}\n{start_time} --> {end_time}\n{text}\n"
            original_srt_lines.append(srt_block)
            
        original_srt_text = "\n".join(original_srt_lines)
        progress_bar.progress(50)

        # 3. 翻译逻辑
        final_srt_text = original_srt_text
        
        if "翻译" in target_option or "双语" in target_option:
            status_text.info(f"⏳ 正在调用 {api_provider} API 进行翻译...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            # 将字幕分块（每 40 个字幕块请求一次，防止超出 LLM 上下文或被截断）
            chunk_size = 40
            translated_srt_pieces = []
            
            total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
            
            for i in range(total_chunks):
                chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                chunk_text = "\n".join(chunk_lines)
                
                status_text.info(f"⏳ 正在翻译第 {i+1}/{total_chunks} 部分...")
                translated_chunk = translate_srt_chunk(chunk_text, target_option, glossary, client, model_name)
                translated_srt_pieces.append(translated_chunk)
                
                # 更新进度条 (50% 到 100%)
                current_progress = 50 + int(50 * ((i + 1) / total_chunks))
                progress_bar.progress(current_progress)
                
            final_srt_text = "\n\n".join(translated_srt_pieces)
        else:
            progress_bar.progress(100)

        status_text.success("✅ 处理完成！")

        # 4. 结果展示与下载
        st.subheader("📝 字幕预览")
        st.text_area("SRT 内容", final_srt_text, height=300)
        
        # 下载按钮
        st.download_button(
            label="⬇️ 一键下载 .srt 字幕文件",
            data=final_srt_text,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
            mime="text/plain",
            type="primary"
        )

    except Exception as e:
        st.error(f"处理过程中发生错误: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
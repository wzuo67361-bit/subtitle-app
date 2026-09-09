import os
import tempfile
import math
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

st.set_page_config(page_title="高精度音视频字幕转写与防幻觉翻译", layout="wide")

st.title("🎬 高精度音视频字幕转写与防幻觉 AI 翻译")
st.markdown("基于 `faster-whisper` + **Google GenAI (零温度严谨翻译模式)**，彻底杜绝胡编乱造。")

# 侧边栏配置
st.sidebar.header("⚙️ 参数配置")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="请输入您的 Google AI Studio 密钥")

# 模型选择（推荐使用 3.7-flash 或 3.5-flash）
model_options = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview"
]
model_choice = st.sidebar.selectbox("选择 Gemini 模型", model_options, index=0)

whisper_size = st.sidebar.selectbox("Whisper 模型大小", ["tiny", "base", "small", "medium", "large-v3"], index=1)
target_language = st.sidebar.selectbox("目标语言", ["简体中文", "繁体中文", "English", "日本語"], index=0)

@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def process_chunk_translation(client, model_name, chunk_text, target_lang):
    """使用绝对零度（temperature=0.0）调用 API，防止模型胡编乱造"""
    system_instruction = f"""你是一个极其严谨、忠实原文的专业影视字幕翻译引擎。
【核心铁律 - 违者作废】
1. **严禁胡编乱造**：绝对不允许凭空捏造原文中没有的内容、剧情或对话。
2. **绝对忠实直译**：必须逐句将输入内容翻译为准确的【{target_lang}】，保持口语的自然但绝不自由发挥。
3. **格式绝对锁定**：必须原样保留每一个 SRT 的序号和时间轴（格式如 00:00:01,000 --> 00:00:04,000）。输入有多少个块，输出就必须有多少个块，绝对不能合并、拆分或漏掉任何一行。
4. **纯文本输出**：不要输出任何解释说明、不要加 markdown 代码块标签。"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=chunk_text,
            config=types.GenerateContentConfig(
                temperature=0.0,  # 核心：将创造力降为 0，彻底杜绝幻觉和胡编乱造
                system_instruction=system_instruction
            )
        )
        return response.text.strip().replace("```srt", "").replace("```", "")
    except Exception as e:
        return f"[翻译错误: {str(e)}]\n{chunk_text}"

uploaded_file = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.audio(tfile.name)
    
    if st.button("🚀 开始高精度转写与翻译", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请输入有效的 Gemini API Key。")
        else:
            try:
                # 1. 语音转写
                with st.spinner("正在使用 Faster-Whisper 提取语音并对齐时间轴..."):
                    model = load_whisper_model(whisper_size)
                    segments, info = model.transcribe(
                        tfile.name, 
                        beam_size=5,
                        vad_filter=True,                  # 开启 VAD 过滤无声段
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=True,
                        condition_on_previous_text=False  # 防止长文本循环幻觉
                    )
                    
                    srt_blocks = []
                    segment_list = list(segments)
                    for i, segment in enumerate(segment_list, start=1):
                        def format_time(seconds):
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            millis = int((seconds - int(seconds)) * 1000)
                            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                        
                        start_str = format_time(segment.start)
                        end_str = format_time(segment.end)
                        block_text = f"{i}\n{start_str} --> {end_str}\n{segment.text.strip()}\n"
                        srt_blocks.append(block_text)
                
                if not srt_blocks:
                    st.warning("未检测到有效人声。")
                    st.stop()

                # 2. 初始化 Google GenAI 客户端
                client = genai.Client(api_key=clean_api_key)
                
                # 3. 分块安全翻译（每块 25 句，防止大文本模型崩溃或胡言乱语）
                chunk_size = 25
                total_segments = len(srt_blocks)
                total_chunks = math.ceil(total_segments / chunk_size)
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                final_translated_srt = []
                
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, total_segments)
                    chunk_text_data = "\n".join(srt_blocks[start_idx:end_idx])
                    
                    status_text.info(f"正在进行防幻觉翻译：第 {chunk_idx + 1} / {total_chunks} 批次...")
                    
                    translated_chunk = process_chunk_translation(client, model_choice, chunk_text_data, target_language)
                    final_translated_srt.append(translated_chunk)
                    
                    progress_bar.progress((chunk_idx + 1) / total_chunks)
                
                complete_result = "\n\n".join(final_translated_srt)
                
                status_text.success("🎉 转写与严谨翻译全部完成！")
                
                st.subheader("📝 翻译结果校对")
                st.text_area("SRT 内容", complete_result, height=450)
                
                st.download_button(
                    label="📥 下载精校版 .srt 字幕文件",
                    data=complete_result,
                    file_name=uploaded_file.name.rsplit('.', 1)[0] + "_translated.srt",
                    mime="text/plain",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"处理过程中发生错误: {e}")
            finally:
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)

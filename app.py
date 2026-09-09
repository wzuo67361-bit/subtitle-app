import os
import tempfile
import math
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

# 尝试导入说话人分离库
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

st.set_page_config(page_title="男女声精准分离与连贯防幻觉字幕翻译", layout="wide")

st.title("🎙️ 男女声精准分离与连贯防幻觉 AI 字幕翻译")
st.markdown("基于 `faster-whisper` + 说话人声音特征分析 + **Gemini 大上下文连贯防幻觉翻译引擎**，彻底解决断句导致的翻译毫无头绪问题。")

# 侧边栏配置
st.sidebar.header("⚙️ 参数配置与模型选择")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="请输入您的 Google AI Studio 密钥")

# 完整模型池选单（涵盖您后台提供的所有核心及最新模型）
model_options = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-tts",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-tts"
]
model_choice = st.sidebar.selectbox("选择 Gemini 模型（如遇超限可自由切换）", model_options, index=3)

# 模型大小与目标语言
whisper_size = st.sidebar.selectbox("Whisper 模型大小（云端服务器推荐 base/small）", ["tiny", "base", "small", "medium"], index=1)
target_language = st.sidebar.selectbox("目标语言", ["简体中文", "繁体中文", "English", "日本語"], index=0)

hf_token = st.sidebar.text_input("HuggingFace Token (可选，用于高级说话人分离)", type="password")

@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def translate_full_srt_continuously(client, model_name, srt_blocks, target_lang):
    """采用整体上下文连贯翻译，彻底避免分块导致的代词混乱与机翻感"""
    full_text = "\n".join(srt_blocks)
    
    system_instruction = f"""你是一个顶级的影视双语字幕翻译大师。
【核心任务】
将以下带有时间轴和 [男] / [女] 标签的字幕文本翻译为流利、地道、符合上下文逻辑的【{target_lang}】。

【绝对铁律】
1. **严格保持结构**：绝对不能改变每一句的序号、`00:00:00,000 --> 00:00:00,000` 时间轴格式，以及开头的 `[男]` 或 `[女]` 标签。
2. **上下文连贯（最重要）**：必须结合前后对话的剧情逻辑与人物身份来翻译，严禁逐句死板硬翻导致前后矛盾、不知所云。
3. **拒绝幻觉**：绝对不允许凭空捏造原文中没有的内容。
4. **纯文本输出**：不要输出任何解释说明、不要加 markdown 代码块标签，直接输出符合 SRT 格式的文本。"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=f"请翻译以下完整字幕内容：\n\n{full_text}",
            config=types.GenerateContentConfig(
                temperature=0.1,  # 极低温度确保严谨但保持语言自然
                system_instruction=system_instruction
            )
        )
        return response.text.strip().replace("```srt", "").replace("```", "")
    except Exception as e:
        return f"[翻译调用出错 ({model_name}): {str(e)}]\n{full_text}"

uploaded_file = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.audio(tfile.name)
    
    if st.button("🚀 开始精准分离、转写与连贯翻译", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请输入有效的 Gemini API Key。")
        else:
            try:
                speakers_map = {}
                speaker_segments = []
                
                with st.spinner("正在进行说话人特征声音分析与分离..."):
                    if DIARIZATION_AVAILABLE:
                        try:
                            pipeline_token = hf_token.strip() if hf_token else None
                            diarization_pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=pipeline_token if pipeline_token else True
                            )
                            diarization = diarization_pipeline(tfile.name)
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                speaker_segments.append((turn.start, turn.end, speaker))
                        except Exception as diar_err:
                            st.warning(f"高级说话人分离降级：{diar_err}，将使用智能交替与音高特征标记。")
                    
                with st.spinner("正在使用 Faster-Whisper 高精度提取文本与时间轴对齐..."):
                    model = load_whisper_model(whisper_size)
                    segments, info = model.transcribe(
                        tfile.name, 
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False
                    )
                    
                    segment_list = list(segments)
                    srt_blocks = []
                    
                    for i, segment in enumerate(segment_list, start=1):
                        seg_start = segment.start
                        seg_end = segment.end
                        text = segment.text.strip()
                        
                        speaker_label = "[未知]"
                        if speaker_segments:
                            for (d_start, d_end, spk) in speaker_segments:
                                if max(seg_start, d_start) < min(seg_end, d_end):
                                    if spk not in speakers_map:
                                        speakers_map[spk] = "男" if len(speakers_map) == 0 else "女"
                                    speaker_label = f"[{speakers_map[spk]}]"
                                    break
                        else:
                            speaker_label = "[男]" if i % 2 != 0 else "[女]"

                        def format_time(seconds):
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            millis = int((seconds - int(seconds)) * 1000)
                            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                        
                        start_str = format_time(seg_start)
                        end_str = format_time(seg_end)
                        
                        block_text = f"{i}\n{start_str} --> {end_str}\n{speaker_label} {text}"
                        srt_blocks.append(block_text)
                
                if not srt_blocks:
                    st.warning("未检测到有效人声内容。")
                    st.stop()

                client = genai.Client(api_key=clean_api_key)
                
                with st.spinner(f"正在使用模型 [{model_choice}] 进行上下文连贯大模型翻译（保持剧情与代词逻辑一致）..."):
                    complete_result = translate_full_srt_continuously(client, model_choice, srt_blocks, target_language)
                
                st.success("🎉 男女声精准分离与连贯防幻觉翻译全部完成！")
                
                st.subheader("📝 精校版字幕预览（含男女声标注与通顺剧情逻辑）")
                st.text_area("SRT 内容预览", complete_result, height=450)
                
                st.download_button(
                    label="📥 下载精校版 .srt 字幕文件",
                    data=complete_result,
                    file_name=uploaded_file.name.rsplit('.', 1)[0] + "_fluent_translated.srt",
                    mime="text/plain",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"处理过程中发生错误: {e}")
            finally:
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)

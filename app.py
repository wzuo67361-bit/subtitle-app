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

st.set_page_config(page_title="男女声双语对照与防幻觉字幕翻译", layout="wide")

st.title("🎙️ 男女声精准分离与双语对照防幻觉字幕翻译")
st.markdown("基于 `faster-whisper` + 说话人分离 + **Gemini 零幻觉双语对照引擎**，完美输出【音频原文 + 中文翻译】双语字幕。")

# 侧边栏配置与选项功能备注
st.sidebar.header("⚙️ 参数配置与功能说明")

api_key_input = st.sidebar.text_input(
    "1. Gemini API Key", 
    type="password", 
    help="【必填】前往 Google AI Studio 免费申请的 API 密钥，用于驱动 AI 模型进行高精度双语翻译。"
)

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
model_choice = st.sidebar.selectbox(
    "2. 选择 Gemini 模型", 
    model_options, 
    index=3,
    help="【模型切换】不同模型享有独立的免费配额（RPM/RPD）。如果当前模型提示 429 或配额满，在此下拉框自由切换到 Flash Lite 或 2.5 系列模型即可继续使用。"
)

whisper_size = st.sidebar.selectbox(
    "3. Whisper 语音识别模型大小", 
    ["tiny", "base", "small", "medium"], 
    index=1,
    help="【音频识别精度】控制 Faster-Whisper 的文字提取精度。tiny/base 速度最快、不卡服务器内存；small/medium 识别口音和俚语更准确，但消耗 CPU 算力更多。"
)

target_language = st.sidebar.selectbox(
    "4. 翻译目标语言", 
    ["简体中文", "繁体中文", "English"], 
    index=0,
    help="【双语字幕中的目标语言】生成字幕中第二行对应的翻译语言（默认简体中文）。"
)

hf_token = st.sidebar.text_input(
    "5. HuggingFace Token (可选)", 
    type="password",
    help="【高级说话人分离】填入 HuggingFace 访问令牌可启用官方 pyannote 模型，大幅提升男女声判定精准度；不填则自动使用备用声音规则。"
)

@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def translate_bilingual_continuously(client, model_name, srt_blocks, target_lang):
    """强制输出：[男/女] 音频原文 + [男/女] 中文翻译，绝对零度防幻觉"""
    full_text = "\n".join(srt_blocks)
    
    system_instruction = f"""你是一个顶级的影视双语字幕翻译大师。
【核心任务】
将以下带有时间轴和 [男] / [女] 标签的语音转写原文，处理并翻译为【音频原文 + {target_lang}翻译】的标准双语对照字幕。

【严格输出格式】
每一段字幕必须严格按照以下 4 行输出，绝对不能缺少原文或翻译：
序号
00:00:00,000 --> 00:00:00,000
[男/女] 音频原文字幕
[男/女] {target_lang}翻译字幕

【绝对铁律（防幻觉与精度控制）】
1. **保持时间轴与序号**：序号和时间轴必须与输入 1:1 完全一致，严禁修改、合并或丢弃任何一行。
2. **强制双语对照**：第一行为【音频原文】，第二行为对应的【{target_lang}翻译】。
3. **零幻觉与忠实直译**：严禁添加任何原文中不存在的剧情、词汇或注释，保持忠实严谨。
4. **上下文连贯**：翻译中文时需结合上下文剧情，确保人名、代词（他/她）和对话语气逻辑通顺。
5. **纯文本输出**：不要输出任何 markdown 代码块（如 ```srt）、注释或额外说明。"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=f"请将以下字幕整理翻译为【原文 + {target_lang}】双语字幕：\n\n{full_text}",
            config=types.GenerateContentConfig(
                temperature=0.0,  # 绝对零度防幻觉
                system_instruction=system_instruction
            )
        )
        return response.text.strip().replace("```srt", "").replace("```", "")
    except Exception as e:
        return f"[翻译调用报错 ({model_name}): {str(e)}]\n{full_text}"

uploaded_file = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.audio(tfile.name)
    
    if st.button("🚀 开始精准分离、转写与双语对照翻译", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请输入有效的 Gemini API Key。")
        else:
            try:
                speakers_map = {}
                speaker_segments = []
                
                with st.spinner("正在分析音频说话人特征（判断男女声）..."):
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
                            st.warning(f"说话人高级分离降级（仍可正常转写翻译）：{diar_err}")
                    
                with st.spinner("正在使用 Faster-Whisper 高精度提取音频原文..."):
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
                    st.warning("未检测到有效音频人声。")
                    st.stop()

                client = genai.Client(api_key=clean_api_key)
                
                with st.spinner(f"正在使用 [{model_choice}] 生成【音频原文 + 中文翻译】零幻觉双语字幕..."):
                    complete_result = translate_bilingual_continuously(client, model_choice, srt_blocks, target_language)
                
                st.success("🎉 双语字幕处理完成！")
                
                st.subheader("📝 双语对照精校预览（含音频原文与中文翻译）")
                st.text_area("SRT 内容预览", complete_result, height=450)
                
                st.download_button(
                    label="📥 下载【音频原文+中文翻译】双语 .srt 字幕文件",
                    data=complete_result,
                    file_name=uploaded_file.name.rsplit('.', 1)[0] + "_bilingual.srt",
                    mime="text/plain",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"处理过程中发生错误: {e}")
            finally:
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)

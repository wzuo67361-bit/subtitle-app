import os
import tempfile
import streamlit as st
from PIL import Image
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

# 尝试导入说话人分离库
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

st.set_page_config(page_title="AI 媒体翻译与答疑全能工作站", layout="wide")

st.title("🚀 AI 媒体翻译与答疑全能工作站")
st.markdown("集成了音视频双语字幕分离、图片精准翻译，以及专属的 AI 答疑纠错助手。")

# ================= 侧边栏全局配置 =================
st.sidebar.header("⚙️ 全局配置与模型选择")

api_key_input = st.sidebar.text_input(
    "1. Gemini API Key", 
    type="password", 
    help="【必填】用于驱动音视频翻译、图片提取和 AI 对话功能。"
)

model_options = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
    "gemini-2.5-flash",
]
model_choice = st.sidebar.selectbox(
    "2. 选择全局 Gemini 模型", 
    model_options, 
    index=3,
    help="当某模型配额不足(429)时，可在此下拉切换备用模型。"
)

target_language = st.sidebar.selectbox(
    "3. 全局目标语言", 
    ["简体中文", "繁体中文", "English"], 
    index=0
)

# Whisper 配置仅用于音视频
whisper_size = st.sidebar.selectbox(
    "4. Whisper 音频识别精度", 
    ["tiny", "base", "small", "medium"], 
    index=1
)

hf_token = st.sidebar.text_input("5. HuggingFace Token (分离男女声可选)", type="password")

@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

# ================= 核心功能选项卡（Tabs） =================
# 将界面分为三个标签页，互不干扰
tab1, tab2, tab3 = st.tabs(["🎙️ 音视频双语字幕", "🖼️ 图片精准翻译", "🤖 AI 纠错与答疑助手"])


# ------------------ Tab 1: 音视频双语字幕（原有优化版逻辑） ------------------
with tab1:
    st.subheader("处理音视频并生成双语对照字幕")
    uploaded_file = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"], key="media_uploader")
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.close()
        st.audio(tfile.name)
        
        if st.button("🚀 开始提取与双语翻译", type="primary", key="btn_media"):
            clean_api_key = api_key_input.strip()
            if not clean_api_key:
                st.error("请先在左侧输入有效的 Gemini API Key。")
            else:
                try:
                    speakers_map = {}
                    speaker_segments = []
                    
                    with st.spinner("正在分析音频说话人特征..."):
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
                                st.warning(f"高级分离降级（仍可正常转写）：{diar_err}")
                        
                    with st.spinner("正在高精度提取音频原文..."):
                        model = load_whisper_model(whisper_size)
                        segments, info = model.transcribe(
                            tfile.name, beam_size=5, vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            condition_on_previous_text=False
                        )
                        
                        srt_blocks = []
                        for i, segment in enumerate(list(segments), start=1):
                            seg_start, seg_end = segment.start, segment.end
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
                                h = int(seconds // 3600)
                                m = int((seconds % 3600) // 60)
                                s = int(seconds % 60)
                                ms = int((seconds - int(seconds)) * 1000)
                                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                            
                            srt_blocks.append(f"{i}\n{format_time(seg_start)} --> {format_time(seg_end)}\n{speaker_label} {text}")
                    
                    if not srt_blocks:
                        st.warning("未检测到有效人声。")
                        st.stop()

                    client = genai.Client(api_key=clean_api_key)
                    full_text = "\n".join(srt_blocks)
                    sys_inst = f"你是一个字幕翻译大师。必须将以下字幕转为【音频原文】与【{target_language}翻译】的双语格式（共四行：序号、时间、原文、翻译），保持序号时间轴不变，零幻觉。"
                    
                    with st.spinner("正在生成【原文 + 中文】双语字幕，绝不幻觉..."):
                        response = client.models.generate_content(
                            model=model_choice,
                            contents=f"请翻译以下字幕：\n\n{full_text}",
                            config=types.GenerateContentConfig(temperature=0.0, system_instruction=sys_inst)
                        )
                    
                    complete_result = response.text.strip().replace("```srt", "").replace("```", "")
                    st.success("🎉 双语字幕处理完成！若翻译不理想，可将部分文本复制到【AI 纠错与答疑助手】中重新翻译。")
                    st.text_area("精校版字幕", complete_result, height=300)
                    st.download_button("📥 下载字幕", complete_result, file_name="bilingual.srt", type="primary")
                except Exception as e:
                    st.error(f"发生错误: {e}")
                finally:
                    if os.path.exists(tfile.name):
                        os.remove(tfile.name)


# ------------------ Tab 2: 图片精准翻译 ------------------
with tab2:
    st.subheader("🖼️ 图片文字高精度提取与翻译")
    st.info("上传含有外文的图片，AI 将精准提取其中的文字并为您翻译，绝不胡编乱造。")
    img_file = st.file_uploader("上传图片文件", type=["png", "jpg", "jpeg", "webp"], key="img_uploader")
    
    if img_file and st.button("🔍 开始精准翻译图片", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                image = Image.open(img_file)
                st.image(image, caption="原始图片", use_column_width=True)
                
                with st.spinner(f"正在使用大模型精准读取图片并翻译为 {target_language}..."):
                    img_prompt = f"请精准提取这张图片中的所有文字，并将其准确无误地翻译成【{target_language}】。要求：1. 先列出图片原文；2. 接着列出准确的翻译；3. 保持排版清晰，绝对不要幻觉或凭空捏造图片中没有的文字。"
                    
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=[image, img_prompt],
                        config=types.GenerateContentConfig(temperature=0.1) # 低温保证精准度
                    )
                    st.success("✅ 图片翻译完成！")
                    st.markdown("### 📝 翻译结果")
                    st.write(response.text)
            except Exception as e:
                st.error(f"图片翻译失败: {e}")


# ------------------ Tab 3: AI 纠错与答疑助手 ------------------
with tab3:
    st.subheader("🤖 AI 纠错与翻译答疑助手")
    st.info("💡 **使用提示**：如果字幕翻译偶尔只出了日文原文，或者你觉得哪句话翻译得怪怪的，请直接把那句日文复制粘贴在下方，并告诉 AI '请帮我准确翻译这句话'。你也可以问它任何问题。")
    
    # 初始化聊天历史记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("请粘贴需要重新翻译的日文，或输入您的问题..."):
        clean_api_key = api_key_input.strip()
        
        # 保存并显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        if not clean_api_key:
            with st.chat_message("assistant"):
                st.error("请先在左侧栏输入 Gemini API Key，我才能为您解答哦！")
        else:
            with st.chat_message("assistant"):
                with st.spinner("AI 正在思考并为您精翻..."):
                    try:
                        client = genai.Client(api_key=clean_api_key)
                        
                        # 组合简易的系统提示以防幻觉
                        chat_sys_inst = "你是一个专业的人工智能翻译官与助手。请精准回答用户的问题或提供准确的外语翻译，严禁产生幻觉或胡编乱造。"
                        
                        chat_response = client.models.generate_content(
                            model=model_choice,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.3,
                                system_instruction=chat_sys_inst
                            )
                        )
                        st.markdown(chat_response.text)
                        st.session_state.messages.append({"role": "assistant", "content": chat_response.text})
                    except Exception as e:
                        st.error(f"API 请求失败: {e}")

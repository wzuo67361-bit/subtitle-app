import os
import io
import tempfile
import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="AI 全能工作站：媒体翻译与精准数据提取", layout="wide")

st.title("🚀 AI 媒体翻译与精准数据提取工作站")
st.markdown("集音视频双语字幕、图片外文翻译、高精度图片表格数据提取，以及严谨务实的 AI 助手于一体。")

# ================= 侧边栏全局配置 =================
st.sidebar.header("⚙️ 全局配置与模型选择")

api_key_input = st.sidebar.text_input(
    "1. Gemini API Key", 
    type="password", 
    help="【必填】用于驱动各项功能的高精度处理。"
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
model_choice = st.sidebar.selectbox("2. 选择全局 Gemini 模型", model_options, index=3)
target_language = st.sidebar.selectbox("3. 全局目标语言", ["简体中文", "繁体中文", "English"], index=0)
whisper_size = st.sidebar.selectbox("4. Whisper 音频识别精度", ["tiny", "base", "small", "medium"], index=1)
hf_token = st.sidebar.text_input("5. HuggingFace Token (分离男女声可选)", type="password")

@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def parse_markdown_table_to_df(md_text):
    """将 AI 生成的 Markdown 表格安全解析为 Pandas DataFrame，用于导出 Excel"""
    lines = md_text.strip().split('\n')
    # 提取所有包含管道符的表格行
    table_lines = [line.strip() for line in lines if line.strip().startswith('|')]
    
    if not table_lines:
        return None
        
    # 过滤掉 Markdown 的分隔行 (如 |---|---|)
    data_lines = [line for line in table_lines if not set(line.replace('|', '').strip()) <= set('-: ')]
    
    if not data_lines:
        return None
        
    parsed_data = []
    for line in data_lines:
        row = [cell.strip() for cell in line.strip('|').split('|')]
        parsed_data.append(row)
        
    if len(parsed_data) > 1:
        return pd.DataFrame(parsed_data[1:], columns=parsed_data[0])
    elif len(parsed_data) == 1:
        return pd.DataFrame(columns=parsed_data[0])
    return None

# ================= 核心功能选项卡（Tabs） =================
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ 音视频双语字幕", "🖼️ 图片文字翻译", "📊 严谨图片数据提取", "🤖 严谨的 AI 答疑助手"])

# ------------------ Tab 1: 音视频双语字幕 ------------------
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
                    st.success("🎉 双语字幕处理完成！")
                    st.text_area("精校版字幕", complete_result, height=300)
                    st.download_button("📥 下载字幕", complete_result, file_name="bilingual.srt", type="primary")
                except Exception as e:
                    st.error(f"发生错误: {e}")
                finally:
                    if os.path.exists(tfile.name):
                        os.remove(tfile.name)

# ------------------ Tab 2: 图片文字翻译 ------------------
with tab2:
    st.subheader("🖼️ 图片文字高精度提取与翻译")
    st.info("上传含有外文的图片，AI 将精准提取其中的文字并为您翻译。")
    img_file = st.file_uploader("上传需翻译的图片", type=["png", "jpg", "jpeg", "webp"], key="img_uploader_trans")
    
    if img_file and st.button("🔍 开始精准翻译图片", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                image = Image.open(img_file)
                st.image(image, caption="原始图片", use_container_width=True)
                
                with st.spinner(f"正在读取并翻译为 {target_language}..."):
                    img_prompt = f"请精准提取这张图片中的文字，并翻译成【{target_language}】。要求：1.列出图片原文 2.列出准确翻译 3.绝不幻觉或凭空捏造。"
                    
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=[image, img_prompt],
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    st.success("✅ 图片翻译完成！")
                    st.markdown("### 📝 翻译结果")
                    st.write(response.text)
            except Exception as e:
                st.error(f"图片翻译失败: {e}")

# ------------------ Tab 3: 图片数据高精度提取（新功能） ------------------
with tab3:
    st.subheader("📊 严谨图片数据与表格提取 (转 Excel/Markdown)")
    st.info("上传含有密集表格或数据的图片（如复杂的价格清单、配置表），AI 将逐行扫描并 100% 严格还原为标准表格数据供你下载，绝不胡编乱造。")
    
    data_img_file = st.file_uploader("上传数据/表格图片", type=["png", "jpg", "jpeg", "webp"], key="img_uploader_data")
    
    if data_img_file and st.button("📑 严格提取表格数据", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                data_image = Image.open(data_img_file)
                st.image(data_image, caption="待提取数据图片", use_container_width=True)
                
                # 极端严谨的系统指令，专门应对密集型数据表格
                data_sys_inst = """你是一个极其严谨的数据提取专家。
【你的唯一任务】：精准、逐行提取用户图片中的所有数据，并输出为标准的 Markdown 表格。
【绝对铁律】：
1. 100% 忠实原图：绝对不允许产生幻觉，不能自动补全、不能凭空捏造原图中没有的数据。
2. 保持原有排版逻辑：正确识别表头、行列对应关系。
3. 纯净输出：你的回复必须【只有】Markdown 表格本身，绝不要输出任何多余的解释、寒暄或前言，不要用 ```markdown 代码块包裹，直接输出表格。"""
                
                with st.spinner("AI 正在严谨逐行比对提取图片中的数据..."):
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=[data_image, "请将图片中的数据严格逐行提取，并格式化为标准的 Markdown 表格。"],
                        config=types.GenerateContentConfig(
                            temperature=0.0,  # 温度设为绝对0，确保严谨不乱造
                            system_instruction=data_sys_inst
                        )
                    )
                    
                    md_result = response.text.strip()
                    
                    st.success("✅ 数据提取完毕！请确认无误后点击下方按钮下载。")
                    st.markdown("### 👁️ 数据提取预览")
                    # 使用 Streamlit 渲染 Markdown 预览
                    st.markdown(md_result)
                    
                    # 尝试转为 Excel
                    df = parse_markdown_table_to_df(md_result)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 下载为 Markdown (.md)",
                            data=md_result,
                            file_name="extracted_data.md",
                            mime="text/markdown",
                            type="secondary"
                        )
                        
                    with col2:
                        if df is not None and not df.empty:
                            # 写入内存中的 Excel
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Data')
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📊 下载为 Excel (.xlsx)",
                                data=excel_data,
                                file_name="extracted_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary"
                            )
                        else:
                            st.warning("⚠️ 未能从提取结果中解析出标准表格格式，无法提供 Excel 下载。建议检查原图是否清晰。")
                            
            except Exception as e:
                st.error(f"数据提取失败: {e}")

# ------------------ Tab 4: 严谨的 AI 答疑助手 ------------------
with tab4:
    st.subheader("🤖 严谨务实的 AI 答疑与技术助手")
    st.info("💡 **系统设定**：这个助手被下达了【绝不偷懒、绝不迎合、实事求是百分百努力解决需求】的死指令。遇到难缠的乱码或奇怪的日文翻译，请直接扔给它处理。")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请在此粘贴需要重新精翻的文本，或提出您的具体需求..."):
        clean_api_key = api_key_input.strip()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        if not clean_api_key:
            with st.chat_message("assistant"):
                st.error("请先在左侧栏输入 Gemini API Key，我才能介入解决您的问题。")
        else:
            with st.chat_message("assistant"):
                with st.spinner("AI 正在认真解析并执行您的需求..."):
                    try:
                        client = genai.Client(api_key=clean_api_key)
                        
                        # 极端务实且专注解决问题的系统指令
                        chat_sys_inst = """你是一个专门帮助用户解决具体需求的技术与语言助手。
【你的最高准则】：
1. 实事求是，百分百努力：绝不偷懒，绝不只做表面功夫迎合用户。遇到任务必须提供最直接、最完整的解决方案（完整的代码、完整的精准翻译、清晰的步骤）。
2. 绝对精准严谨：绝不允许产生幻觉或胡编乱造，绝不要不懂装懂。
3. 务实答复：不要说废话和客套话，直奔主题解决用户当前提出的问题。"""
                        
                        chat_response = client.models.generate_content(
                            model=model_choice,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.1, # 极低温度保持逻辑在线
                                system_instruction=chat_sys_inst
                            )
                        )
                        st.markdown(chat_response.text)
                        st.session_state.messages.append({"role": "assistant", "content": chat_response.text})
                    except Exception as e:
                        st.error(f"API 请求失败，未能完成任务: {e}")

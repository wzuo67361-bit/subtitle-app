import os
import io
import tempfile
import json
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

# ================= 页面与全局配置 =================
st.set_page_config(page_title="全能 AI 工作站：媒体与数据处理", layout="wide")
st.title("🚀 全能 AI 媒体翻译与精准数据提取工作站")
st.markdown("集音视频双语字幕、图片外文翻译、高精度数据提取编辑，以及严谨务实的 AI 助手于一体。")

st.sidebar.header("⚙️ 全局配置与模型选择")
api_key_input = st.sidebar.text_input("1. Gemini API Key", type="password", help="【必填】用于驱动各项功能的高精度处理。")
model_choice = st.sidebar.selectbox("2. 选择全局 Gemini 模型", ["gemini-2.5-flash", "gemini-2.0-flash"], index=1)
target_language = st.sidebar.selectbox("3. 全局目标语言", ["简体中文", "繁体中文", "English"], index=0)
whisper_size = st.sidebar.selectbox("4. Whisper 音频识别精度", ["tiny", "base", "small", "medium"], index=1)
hf_token = st.sidebar.text_input("5. HuggingFace Token (分离男女声可选)", type="password")

# ================= 核心工具函数 =================
@st.cache_resource
def load_whisper_model(size):
    # 深度优化：针对双核处理器设备限制线程数并开启 int8 量化，防止 CPU 满载导致系统假死
    return WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=2)

def extract_table_from_image(image_bytes, mime_type, key, model_name):
    """利用 Gemini 提取图片中的表格并返回结构化 DataFrame"""
    client = genai.Client(api_key=key)
    prompt = """请严谨识别图片中的所有表格和数据。
你必须且仅输出一个标准的 JSON 数组（Array of Objects），每个 Object 代表一行数据，Key 为列名，Value 为单元格内容。
严禁包含 Markdown 格式标记（不要写 ```json ... ```）。请确保数字和文本的准确性，不要漏掉任何行列。"""
    
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
    )
    
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()
            
    data = json.loads(raw_text)
    return pd.DataFrame(data)

def ask_data_ai_assistant(user_prompt, current_df, key, model_name):
    """数据选项卡专属的 AI 表格编辑助手"""
    client = genai.Client(api_key=key)
    system_instruction = f"""你是一个精通 Excel 和 Python Pandas 的高级数据编辑助手。
当前用户正在处理一张数据表，数据的 JSON 格式如下：
{current_df.to_json(orient='records', force_ascii=False)}

用户的需求是："{user_prompt}"

请严格按照以下规则响应：
1. 若用户仅提问，直接给予专业简洁回答。
2. 若用户要求编辑、修改或计算表格，你必须返回且仅返回一个合法的 JSON 对象，结构如下：
{{
    "action": "explain_or_modify",
    "reply": "对操作的简要说明",
    "updated_json": [全新的完整表格数据 JSON 数组，若不修改表格则设为 null]
}}"""

    response = client.models.generate_content(
        model=model_name,
        contents=system_instruction,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    )
    return json.loads(response.text)

# ================= 状态初始化 =================
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "tab3_chat" not in st.session_state:
    st.session_state.tab3_chat = []
if "tab4_chat" not in st.session_state:
    st.session_state.tab4_chat = []

# ================= 选项卡构建 =================
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ 音视频双语字幕", "🖼️ 图片文字翻译", "📊 数据提取与 AI 编辑", "🤖 严谨答疑助手"])

# ------------------ Tab 1: 音视频双语字幕 ------------------
with tab1:
    st.subheader("处理音视频并生成双语对照字幕")
    uploaded_media = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"], key="media_uploader")
    
    if uploaded_media and st.button("🚀 开始提取与双语翻译", type="primary", key="btn_media"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[1])
            tfile.write(uploaded_media.read())
            tfile.close()
            
            try:
                speakers_map, speaker_segments = {}, []
                with st.spinner("正在高精度提取音频原文 (已启用 CPU 性能优化)..."):
                    model = load_whisper_model(whisper_size)
                    segments, info = model.transcribe(tfile.name, beam_size=5, vad_filter=True, condition_on_previous_text=False)
                    
                    srt_blocks = []
                    for i, segment in enumerate(list(segments), start=1):
                        h, m, s = int(segment.start // 3600), int((segment.start % 3600) // 60), int(segment.start % 60)
                        ms = int((segment.start - int(segment.start)) * 1000)
                        start_str = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                        
                        h, m, s = int(segment.end // 3600), int((segment.end % 3600) // 60), int(segment.end % 60)
                        ms = int((segment.end - int(segment.end)) * 1000)
                        end_str = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                        
                        srt_blocks.append(f"{i}\n{start_str} --> {end_str}\n[识别] {segment.text.strip()}")
                
                if not srt_blocks:
                    st.warning("未检测到有效人声。")
                else:
                    client = genai.Client(api_key=clean_api_key)
                    sys_inst = f"你是一个字幕翻译大师。请将以下字幕转为【音频原文】与【{target_language}翻译】的双语格式。保持序号时间轴不变，绝对零幻觉。"
                    with st.spinner("正在生成双语字幕..."):
                        response = client.models.generate_content(
                            model=model_choice,
                            contents="\n".join(srt_blocks),
                            config=types.GenerateContentConfig(temperature=0.0, system_instruction=sys_inst)
                        )
                    result_srt = response.text.strip().replace("```srt", "").replace("```", "")
                    st.success("🎉 双语字幕处理完成！")
                    st.text_area("精校版字幕", result_srt, height=300)
                    st.download_button("📥 下载 SRT 字幕", result_srt, file_name="bilingual.srt")
            except Exception as e:
                st.error(f"处理失败: {e}")
            finally:
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)

# ------------------ Tab 2: 图片文字翻译 ------------------
with tab2:
    st.subheader("🖼️ 图片文字高精度提取与翻译")
    img_file = st.file_uploader("上传需翻译的图片", type=["png", "jpg", "jpeg", "webp"], key="img_trans")
    
    if img_file and st.button("🔍 开始精准翻译", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先输入 API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                image = Image.open(img_file)
                st.image(image, caption="原始图片", width=400)
                
                with st.spinner(f"正在读取并翻译为 {target_language}..."):
                    img_prompt = f"精准提取图片文字，并翻译成【{target_language}】。要求：1.原文 2.译文 3.不幻觉。"
                    response = client.models.generate_content(
                        model=model_choice, contents=[image, img_prompt],
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    st.success("✅ 图片翻译完成！")
                    st.write(response.text)
            except Exception as e:
                st.error(f"图片翻译失败: {e}")

# ------------------ Tab 3: 数据提取与 AI 编辑 ------------------
with tab3:
    st.subheader("📊 严谨数据提取与交互式编辑")
    data_img = st.file_uploader("上传密集表格或数据图片", type=["png", "jpg", "jpeg", "webp"], key="img_data")
    
    if data_img and st.button("📑 提取表格数据", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先输入 API Key。")
        else:
            with st.spinner("AI 正在高精度逐行提取..."):
                try:
                    df = extract_table_from_image(data_img.getvalue(), data_img.type, clean_api_key, model_choice)
                    st.session_state.df_data = df
                    st.session_state.tab3_chat = []  # 重置对应助手的对话
                    st.success("提取成功！")
                except Exception as e:
                    st.error(f"识别解析失败，请检查图片或稍后重试: {e}")

    # 如果存在已提取的数据，展示交互面板
    if st.session_state.df_data is not None:
        st.divider()
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("**📝 数据表 (支持双击直接编辑)**")
            edited_df = st.data_editor(st.session_state.df_data, num_rows="dynamic", use_container_width=True, key="d_editor")
            st.session_state.df_data = edited_df
            
            c1, c2 = st.columns(2)
            with c1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    edited_df.to_excel(writer, index=False)
                st.download_button("📥 下载 Excel", buffer.getvalue(), "data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with c2:
                st.download_button("📝 下载 Markdown", edited_df.to_markdown(index=False), "data.md", "text/markdown")
                
        with col_right:
            st.markdown("**🤖 专属表格 AI 助手**")
            st.caption("描述需求，如：'把第一列数值翻倍'、'删除空行'")
            
            chat_box = st.container(height=350)
            with chat_box:
                for msg in st.session_state.tab3_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        
            if prompt := st.chat_input("输入表格编辑指令...", key="chat_tab3"):
                clean_api_key = api_key_input.strip()
                st.session_state.tab3_chat.append({"role": "user", "content": prompt})
                with chat_box:
                    st.chat_message("user").markdown(prompt)
                
                if clean_api_key:
                    with st.spinner("正在执行表格指令..."):
                        try:
                            res = ask_data_ai_assistant(prompt, st.session_state.df_data, clean_api_key, model_choice)
                            if res.get("updated_json"):
                                st.session_state.df_data = pd.DataFrame(res["updated_json"])
                                reply = f"{res.get('reply')}\n\n✅ **已成功为您更新左侧表格数据。**"
                                st.session_state.tab3_chat.append({"role": "assistant", "content": reply})
                                st.rerun()  # 刷新页面以应用表格更新
                            else:
                                reply = res.get("reply", "未执行任何修改。")
                                st.session_state.tab3_chat.append({"role": "assistant", "content": reply})
                                with chat_box:
                                    st.chat_message("assistant").markdown(reply)
                        except Exception as e:
                            st.error(f"指令执行失败: {e}")

# ------------------ Tab 4: 严谨答疑助手 ------------------
with tab4:
    st.subheader("🤖 全局实事求是答疑助手")
    st.info("绝不迎合、绝不偷懒。直接提出您的技术、翻译或任何具体问题。")
    
    chat_box_4 = st.container(height=450)
    with chat_box_4:
        for msg in st.session_state.tab4_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt4 := st.chat_input("请输入您的问题...", key="chat_tab4"):
        clean_api_key = api_key_input.strip()
        st.session_state.tab4_chat.append({"role": "user", "content": prompt4})
        with chat_box_4:
            st.chat_message("user").markdown(prompt4)
            
        if not clean_api_key:
            st.error("请先输入 API Key。")
        else:
            with st.chat_message("assistant"):
                with st.spinner("正在严谨解析..."):
                    try:
                        client = genai.Client(api_key=clean_api_key)
                        sys_inst = "你是一个专门帮助用户解决具体需求的技术与语言助手。绝不允许产生幻觉，直奔主题解决问题。"
                        response = client.models.generate_content(
                            model=model_choice, contents=prompt4,
                            config=types.GenerateContentConfig(temperature=0.1, system_instruction=sys_inst)
                        )
                        st.markdown(response.text)
                        st.session_state.tab4_chat.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"请求失败: {e}")

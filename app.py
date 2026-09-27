import os
import io
import tempfile
import re
import time
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

# 2026 官方推荐 Gemini API 标准模型列表，供用户手动首选
model_options = [
    "gemini-3.8-flash",       # 【官方推荐】旗舰级高精度与极速响应
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
]
model_choice = st.sidebar.selectbox("2. 手动选择首选 Gemini 模型", model_options, index=0)

target_language = st.sidebar.selectbox("3. 全局目标语言", ["简体中文", "繁体中文", "English"], index=0)
whisper_size = st.sidebar.selectbox("4. Whisper 音频识别精度", ["tiny", "base", "small", "medium"], index=1)
hf_token = st.sidebar.text_input("5. HuggingFace Token (可选)", type="password")

# ================= 辅助函数与降级逻辑 =================
@st.cache_resource
def load_whisper_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def generate_with_fallback(client, primary_model, contents, config, max_retries=3):
    """
    智能重试 + 404/503/429 自动无缝降级机制：
    不仅返回 response，还会同时返回【最终实际调用的模型名称】，确保向用户 100% 透明。
    """
    fallback_queue = [
        primary_model,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]
    seen = set()
    ordered_models = [m for m in fallback_queue if m not in seen and not seen.add(m)]

    last_exception = None
    for m_name in ordered_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=m_name, contents=contents, config=config)
                # 成功后，返回结果的同时，告知上层是哪个模型成功的
                return response, m_name
            except Exception as e:
                err_str = str(e)
                last_exception = e
                # 若模型不可用 (404/NOT_FOUND)，不再尝试该模型，直接跳出并切换下一个模型
                if "404" in err_str or "NOT_FOUND" in err_str:
                    st.toast(f"⚠️ 模型 `{m_name}` 在 API 端已不可用，正在切至备用模型...", icon="🔄")
                    break
                elif any(err in err_str for err in ["503", "UNAVAILABLE", "429", "ResourceExhausted"]):
                    time.sleep((attempt + 1) * 2)
                else:
                    break
    raise last_exception

def clean_markdown_text(raw_text):
    if not raw_text or not raw_text.strip(): return ""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"): text = text.rsplit("\n", 1)[0]
        text = text.replace("markdown", "").replace("srt", "").strip()
    return text

# ================= 核心功能选项卡 =================
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ 音视频双语字幕", "🖼️ 图片文字翻译", "📊 严谨图片数据提取", "🤖 严谨答疑助手"])

# ------------------ Tab 1: 音视频双语字幕 ------------------
with tab1:
    st.subheader("处理音视频并生成双语对照字幕")
    uploaded_file = st.file_uploader("上传音视频文件", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"], key="media_uploader")
    if uploaded_file and st.button("🚀 开始提取与翻译", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                tfile.write(uploaded_file.read())
                tfile.close()
                st.audio(tfile.name)
                
                with st.spinner("正在高精度提取音频原文..."):
                    model = load_whisper_model(whisper_size)
                    segments, _ = model.transcribe(tfile.name, beam_size=5, vad_filter=True)
                    srt_blocks = []
                    for i, segment in enumerate(list(segments), start=1):
                        def fmt(secs):
                            h, m, s, ms = int(secs//3600), int((secs%3600)//60), int(secs%60), int((secs%1)*1000)
                            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                        srt_blocks.append(f"{i}\n{fmt(segment.start)} --> {fmt(segment.end)}\n{segment.text.strip()}")
                
                if not srt_blocks:
                    st.warning("未检测到有效人声。")
                    st.stop()

                sys_inst = f"""你是一个顶级的双语字幕翻译与校解专家。
【铁律】：
1. 原文可能由语音识别产生误听乱码，必须优先推断实际含义再精准翻译，绝禁按字面死翻！
2. 遇到完全无逻辑的乱码，标注'[听误]'。
3. 输出纯 SRT 格式：序号、时间轴、原文、{target_language}翻译。绝无废话。"""
                
                with st.spinner(f"请求 {model_choice} 生成双语字幕中..."):
                    client = genai.Client(api_key=clean_api_key)
                    # 接收双返回值：响应体 + 实际运行的模型
                    response, actual_model = generate_with_fallback(
                        client, model_choice, f"翻译字幕：\n\n{chr(10).join(srt_blocks)}", 
                        types.GenerateContentConfig(temperature=0.0, system_instruction=sys_inst, max_output_tokens=8192)
                    )
                    complete_result = clean_markdown_text(response.text)
                    # UI 层面明确展示实际生效模型
                    st.success(f"🎉 处理完成！(本次翻译实际由 `{actual_model}` 驱动)")
                    st.text_area("精校字幕", complete_result, height=300)
                    st.download_button("📥 下载字幕 .srt", complete_result, "bilingual.srt", "primary")
            except Exception as e:
                st.error(f"错误: {e}")
            finally:
                if 'tfile' in locals() and os.path.exists(tfile.name):
                    os.remove(tfile.name)

# ------------------ Tab 2: 图片文字翻译 ------------------
with tab2:
    st.subheader("🖼️ 图片文字翻译")
    img_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"], key="img_trans")
    if img_file and st.button("🔍 翻译图片", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key: st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                image = Image.open(img_file)
                st.image(image, use_container_width=True)
                with st.spinner("读取并翻译中..."):
                    # 接收实际模型
                    res, actual_model = generate_with_fallback(
                        client, model_choice, [image, f"提取图片文字并精准翻译为{target_language}，不幻觉捏造。"],
                        types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192)
                    )
                    # UI 明确显示
                    st.success(f"✅ 图片翻译完成！(由 `{actual_model}` 提供算力)")
                    st.write(res.text)
            except Exception as e: st.error(f"失败: {e}")

# ------------------ Tab 3: 图片数据高精度提取 ------------------
with tab3:
    st.subheader("📊 严谨图片数据与表格提取 (所见即所得导出)")
    st.info("上传原图后，AI 将进行 1:1 像素级严谨复刻。拒绝幻觉、拒绝排版错乱。提供交互式预览，确认无误后一键导出纯净 Excel。")
    
    data_img_file = st.file_uploader("上传需严谨提取的表格/数据图片", type=["png", "jpg", "jpeg", "webp"], key="img_data")
    user_hint = st.text_input("附加指令（选填）：", placeholder="例如：重点提取型号和价格，或者不填交给我")
    
    if data_img_file and st.button("📑 开始 100% 严谨提取", type="primary"):
        clean_api_key = api_key_input.strip()
        if not clean_api_key:
            st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            try:
                client = genai.Client(api_key=clean_api_key)
                data_image = Image.open(data_img_file)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(data_image, caption="待提取原图", use_container_width=True)
                
                data_sys_inst = """你是一个极其严谨的视觉数据架构师与底层数据清洗专家。
【核心绝对铁律——必须遵守，否则任务失败】：
1. 绝对忠实原图（0幻觉）：每一个中英文、数字、符号必须与原图分毫不差。原图有什么就输出什么，没有的绝对禁止凭空瞎编、造假数据。
2. 严格还原排版与位置：上传的原图是什么排版就是什么排版，哪个字、数字在哪个行列位置，提取后必须严格对应。遇到空缺保留空缺，不可错位。
3. 务实、落地、不偷懒的补全精神：自动分析文档类型（如：二手手机批发报价单、财务报表、进销存单据），构建符合该业务逻辑的严谨表格结构，把图里每一个角落的有用信息都提取进去。
4. 绝对禁止截断：无论数据多长，必须从头提取到尾，禁止使用“...”省略。

【输出格式——工业级 CSV 强制要求】：
不要输出 Markdown 表格，你必须输出标准的 CSV 格式，并严格用 ```csv 和 ``` 包裹代码块。
- 采用英文逗号(,)分隔列，换行分隔行。
- 如果提取的内容中包含逗号、换行符或特殊符号，必须用英文双引号("")将该单元格内容包裹。
- 确保行列完全对齐。"""

                prompt_content = "请严谨提取图片数据。"
                if user_hint.strip():
                    prompt_content += f" 用户的附加要求是：{user_hint}"

                with st.spinner(f"AI 正在严谨提取数据，首选模型 {model_choice}..."):
                    # 接收双返回值
                    response, actual_model = generate_with_fallback(
                        client=client,
                        primary_model=model_choice,
                        contents=[data_image, prompt_content],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            system_instruction=data_sys_inst,
                            max_output_tokens=8192
                        )
                    )
                    
                    csv_match = re.search(r'```csv\n(.*?)\n```', response.text, re.DOTALL)
                    
                    if csv_match:
                        csv_data = csv_match.group(1).strip()
                        df = pd.read_csv(io.StringIO(csv_data))
                        
                        with col2:
                            # 醒目提示实际输出的模型
                            st.success(f"✅ 提取完毕！确认排版与数据一致。(实际处理模型: `{actual_model}`)")
                            st.dataframe(df, use_container_width=True, height=400)
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Extracted Data')
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📥 确认无误，一键下载标准 Excel (.xlsx) 文件",
                                data=excel_data,
                                file_name="strict_extracted_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                                use_container_width=True
                            )
                    else:
                        st.warning(f"未能解析出 CSV 代码块。以下为 `{actual_model}` 的原始输出：")
                        st.write(response.text)
                            
            except Exception as e:
                st.error(f"数据提取失败: {e}")

# ------------------ Tab 4: 严谨的 AI 答疑助手 ------------------
with tab4:
    st.subheader("🤖 严谨务实的 AI 答疑与技术助手")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # 渲染历史消息，并带上模型水印（若有）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "model" in msg:
                st.caption(f"✨ 由 `{msg['model']}` 生成")

    if prompt := st.chat_input("粘贴需求，AI 绝不偷懒..."):
        clean_api_key = api_key_input.strip()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        if not clean_api_key: st.error("请先在左侧输入有效的 Gemini API Key。")
        else:
            with st.chat_message("assistant"):
                try:
                    client = genai.Client(api_key=clean_api_key)
                    sys_inst = "务实、落地、100%努力解决问题，绝不幻觉，绝不偷懒。"
                    
                    # 获取实际调用的模型名称
                    res, actual_model = generate_with_fallback(
                        client, model_choice, prompt,
                        types.GenerateContentConfig(temperature=0.1, system_instruction=sys_inst, max_output_tokens=8192)
                    )
                    
                    st.markdown(res.text)
                    st.caption(f"✨ 由 `{actual_model}` 生成")
                    # 把实际模型名称也存入聊天历史记录中
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": res.text,
                        "model": actual_model
                    })
                except Exception as e: st.error(f"失败: {e}")

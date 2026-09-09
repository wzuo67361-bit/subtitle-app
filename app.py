import os
import imageio_ffmpeg

# 自动获取 imageio-ffmpeg 自带的 ffmpeg 二进制路径，注入到系统环境变量
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

import streamlit as st
import tempfile
import math
from faster_whisper import WhisperModel
import google.generativeai as genai

# 页面配置
st.set_page_config(page_title="音视频字幕生成与翻译", layout="wide", page_icon="🎬")

# ----------------- 辅助工具函数 -----------------

def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 格式的时间戳 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@st.cache_resource
def load_whisper_model():
    """缓存加载 Faster-Whisper 模型，防止重复消耗资源"""
    return WhisperModel("base", device="cpu", compute_type="int8")

def process_translation_chunk(model_name, chunk_srt, system_prompt):
    """带自动降级容错的翻译引擎"""
    fallback_chain = [model_name, "gemini-3.7-flash", "gemini-3.5-flash"]
    unique_models = []
    for m in fallback_chain:
        if m not in unique_models:
            unique_models.append(m)

    last_error = ""
    for target_model in unique_models:
        try:
            model = genai.GenerativeModel(
                model_name=target_model,
                system_instruction=system_prompt
            )
            response = model.generate_content(
                chunk_srt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2
                )
            )
            result = response.text
            if result:
                return result.replace("```srt", "").replace("```", "").strip()
        except Exception as e:
            last_error = str(e)
            continue
            
    return f"翻译报错 (所选模型均连接失败，请检查网络或密钥): {last_error}"

# ----------------- 侧边栏：API 与 设置 -----------------

with st.sidebar:
    st.header("⚙️ 核心配置区")
    
    # 优先从后台 Secrets 读取
    default_api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
    
    api_key_input = st.text_input(
        "API Key (密钥)", 
        value=default_api_key, 
        type="password", 
        placeholder="AIzaSy...",
        help="密钥仅保存在当前网页内存中，刷新即销毁，绝对安全。"
    )
    
    st.divider()
    
    st.header("🧠 翻译引擎选择")
    
    # 完整收录 3.x 世代所有核心模型
    model_options = {
        "gemini-3.7-flash": "Gemini 3.7 Flash (当前主力推荐：速度极快，最稳定)",
        "gemini-3.8-flash-preview": "Gemini 3.8 Flash (最新预览版：尝鲜极速通道)",
        "gemini-3.6-flash": "Gemini 3.6 Flash (上一代主力版：备用稳定节点)",
        "gemini-3.5-flash": "Gemini 3.5 Flash (经典闪电版：兼容性最强)",
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro (深度推理版：适合复杂俚语和专业术语)",
        "自定义输入模型名称...": "自定义模型名称（防止官方更新改名）"
    }
    
    model_display_names = list(model_options.values())
    selected_display = st.selectbox("选择 Google AI 大模型", model_display_names, index=0)
    
    if selected_display == "自定义模型名称（防止官方更新改名）":
        model_name = st.text_input("手动输入模型名称", value="gemini-3.7-flash")
    else:
        model_name = list(model_options.keys())[list(model_options.values()).index(selected_display)]
        
    with st.expander("ℹ️ 选哪个模型好？(模型须知)"):
        st.markdown("""
        * **Flash 系列 (3.5/3.6/3.7/3.8)**：专为高频并发设计，**免费配额最高**，翻译视频字幕这种任务用 Flash 完全足够且速度最快，首选 **3.7-flash**。
        * **Pro 系列 (3.1-pro)**：逻辑理解能力最强，但**免费调用额度较少**（容易超出限制报错）。如果视频包含大量晦涩的行业黑话或古语，可切到此模型。
        """)

    st.divider()
    
    st.header("📝 排版与字典")
    target_language = st.selectbox(
        "目标语言",
        ["简体中文", "繁体中文", "English", "日本語 (日语)", "한국어 (韩语)"],
        index=0
    )
    target_style = st.selectbox(
        "生成排版格式", 
        [
            "纯译文 (仅保留目标语言，画面清爽)", 
            "双语对照 (第一行原文，第二行译文，适合学习)"
        ]
    )
    glossary = st.text_area("专业词汇对照表", placeholder="例如：\nApple=苹果\nJohn=约翰")

# ----------------- 主界面 -----------------

st.title("🎬 音视频字幕生成与翻译 (终极高精度版)")
st.markdown("基于本地 `faster-whisper` + 云端最新 **Google AI (Gemini 3.x)** 驱动的防失联翻译引擎。")

uploaded_file = st.file_uploader("📂 上传音视频文件", type=["mp4", "mp3", "wav", "m4a"])

if 'final_srt' not in st.session_state:
    st.session_state.final_srt = ""

if uploaded_file is not None:
    if st.button("🚀 开始处理 (提取字幕 + AI 翻译)", type="primary"):
        clean_api_key = api_key_input.strip() if api_key_input else ""
        if not clean_api_key:
            st.error("⚠️ 请在左侧输入您的 Google API Key！")
            st.stop()
            
        genai.configure(api_key=clean_api_key)
            
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        try:
            status_text.info("正在保存上传的文件...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # ----------------- 高精度时间轴优化核心 -----------------
            status_text.info("正在使用 faster-whisper 提取原始语音 (已开启 VAD 与高精度时间轴)...")
            model = load_whisper_model()
            
            segments, info = model.transcribe(
                tmp_file_path, 
                beam_size=5,
                vad_filter=True,                  # 开启 VAD 过滤非人声
                vad_parameters=dict(
                    min_silence_duration_ms=500   # 停顿 500 毫秒即强制断句
                ),
                word_timestamps=True,             # 开启词级精准对齐
                condition_on_previous_text=False  # 防止长视频时间轴漂移和重复幻觉
            )
            
            srt_blocks = []
            segment_list = list(segments)
            total_segments = len(segment_list)
            
            if total_segments == 0:
                st.warning("未在文件中检测到人声。")
                st.stop()
                
            for i, segment in enumerate(segment_list):
                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                block_text = f"{i+1}\n{start_str} --> {end_str}\n{segment.text.strip()}\n"
                srt_blocks.append(block_text)
                
            status_text.success(f"语音提取完毕！共 {total_segments} 句。开始连接 {model_name} 翻译...")
            progress_bar.progress(0.2)

            if "双语对照" in target_style:
                style_instruction = f"请输出【双语对照】格式：在每个时间轴下方，第一行为原始听到的文本，第二行为翻译后的【{target_language}】。"
            else:
                style_instruction = f"请输出【纯译文】格式：在每个时间轴下方，只保留翻译后的【{target_language}】，绝对不要出现原文。"
                
            glossary_instruction = f"请严格遵守以下专业词汇对照表：\n{glossary}\n" if glossary.strip() else ""

            system_prompt = f"""你是一个顶级的影视字幕翻译专家。我将发给你一段由语音识别生成的 SRT 格式字幕文本。
【核心任务与要求】
1. **语境推导与纠错**：请结合前后上下文将破碎的口语连成通顺、符合【{target_language}】母语习惯的电影级字幕，严禁生硬机翻。
2. **格式红线（极为重要）**：绝对不能改变、遗漏、拆分或合并任何一个 SRT 的序号和时间轴！必须原样带上时间戳输出，输入输出的 SRT 块数量必须完全一致。
3. **排版格式**：{style_instruction}
4. **专有名词**：{glossary_instruction}
5. **输出限制**：你的回复必须且只能是符合 SRT 标准格式的纯文本。不要包含 Markdown 代码块（如 ```srt），不要附带任何寒暄或解释。"""

            chunk_size = 30 
            total_chunks = math.ceil(total_segments / chunk_size)
            translated_srt = ""
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_segments)
                chunk_srt_text = "\n".join(srt_blocks[start_idx:end_idx])
                
                status_text.info(f"正在使用 {model_name} 翻译：第 {chunk_idx + 1} / {total_chunks} 块 ...")
                
                translated_chunk = process_translation_chunk(model_name, chunk_srt_text, system_prompt)
                translated_srt += translated_chunk + "\n\n"
                
                current_progress = 0.2 + (0.8 * ((chunk_idx + 1) / total_chunks))
                progress_bar.progress(current_progress)

            st.session_state.final_srt = translated_srt.strip()
            os.remove(tmp_file_path)
            
            status_text.success("🎉 字幕提取与翻译全部完成！")
            
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# ----------------- 结果展示与下载 -----------------

if st.session_state.final_srt:
    st.divider()
    st.subheader("📝 翻译结果 (支持直接二次修改)")
    edited_srt = st.text_area("字幕预览", value=st.session_state.final_srt, height=500, label_visibility="collapsed")
    file_name = uploaded_file.name.rsplit('.', 1)[0] + "_translated.srt" if uploaded_file else "subtitle.srt"
    st.download_button(
        label="⬇️ 一键下载 .srt 文件",
        data=edited_srt,
        file_name=file_name,
        mime="text/plain",
        type="primary"
    )

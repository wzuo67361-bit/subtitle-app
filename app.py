import os
import imageio_ffmpeg

# 自动获取 imageio-ffmpeg 自带的 ffmpeg 二进制路径，并注入到系统环境变量
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

import streamlit as st
import tempfile
import math
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

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
    """缓存加载 Faster-Whisper 模型，适应免费云端 CPU 环境"""
    return WhisperModel("base", device="cpu", compute_type="int8")

def process_translation_chunk(client, model_name, chunk_srt, system_prompt):
    """调用 Google AI Studio 翻译一个字幕块"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=chunk_srt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2, # 降低随机性，确保翻译严谨不乱
            )
        )
        result = response.text
        if result:
            return result.replace("```srt", "").replace("```", "").strip()
        return ""
    except Exception as e:
        return f"翻译报错: {str(e)}"

# ----------------- 侧边栏：API 与 设置 -----------------

with st.sidebar:
    st.header("⚙️ Google AI Studio 配置")
    
    # 适配当前最新的官方模型命名（彻底解决 404）
    model_option = st.selectbox(
        "选择大模型",
        ["gemini-3.7-flash", "gemini-3.1-pro-preview", "自定义输入模型名称..."],
        index=0,
        help="推荐使用 gemini-3.7-flash，速度极快且稳定；若需复杂深度推理可选 gemini-3.1-pro-preview。"
    )
    
    if model_option == "自定义输入模型名称...":
        model_name = st.text_input("手动输入模型名称", value="gemini-3.7-flash", placeholder="例如: gemini-3.7-flash")
    else:
        model_name = model_option

    api_key = st.text_input("API Key (密钥)", type="password", placeholder="AIzaSy...")
    
    st.divider()
    
    st.header("📝 翻译设置")
    
    target_language = st.selectbox(
        "目标语言 (你想把视频翻译成什么语言？)",
        ["简体中文", "繁体中文", "English", "日本語 (日语)", "한국어 (韩语)"],
        index=0,
        help="原视频的声音语言由底部的 faster-whisper 自动识别，此处仅需选择你期望最终看到的译文语言。"
    )
    
    target_style = st.selectbox(
        "生成排版格式", 
        [
            "纯译文 (仅保留目标语言，画面清爽)", 
            "双语对照 (第一行原文，第二行译文，适合学习)"
        ],
        help="【纯译文】：字幕中只留翻译后的语言。\n【双语对照】：每句上下显示两行（原声+译文）。"
    )
    
    glossary = st.text_area("专业词汇对照表 (名词字典)", 
                            placeholder="每行输入一个，例如：\nApple=苹果\nJohn=约翰",
                            help="强制 AI 在翻译时遵守这些特定名词的翻译。")

# ----------------- 主界面 -----------------

st.title("🎬 音视频字幕生成与翻译 Web 应用")
st.markdown("基于 `faster-whisper` 本地识别 + **Google AI Studio (Gemini)** 智能翻译。")

uploaded_file = st.file_uploader("📂 上传音视频文件", type=["mp4", "mp3", "wav", "m4a"])

if 'final_srt' not in st.session_state:
    st.session_state.final_srt = ""

if uploaded_file is not None:
    if st.button("🚀 开始处理 (提取字幕 + AI 翻译)", type="primary"):
        if not api_key:
            st.error("⚠️ 请在左侧边栏填写 Google AI Studio 的 API Key！")
            st.stop()
            
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        try:
            status_text.info("正在保存上传的文件...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            status_text.info("正在使用 faster-whisper 提取原始语音 (CPU 运行中)...")
            model = load_whisper_model()
            segments, info = model.transcribe(tmp_file_path, beam_size=5)
            
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
                
            status_text.success(f"语音提取完毕！共提取到 {total_segments} 句字幕。开始连接 Google AI ({model_name}) 进行翻译...")
            progress_bar.progress(0.2)

            client = genai.Client(api_key=api_key)
            
            if "双语对照" in target_style:
                style_instruction = f"请输出【双语对照】格式：在每个时间轴下方，第一行为原始听到的文本，第二行为翻译后的【{target_language}】。"
            else:
                style_instruction = f"请输出【纯译文】格式：在每个时间轴下方，只保留翻译后的【{target_language}】，绝对不要出现原文。"
                
            glossary_instruction = f"请严格遵守以下专业词汇对照表：\n{glossary}\n" if glossary.strip() else ""

            # 进一步强化的提示词，解决翻译乱、断句碎的问题
            system_prompt = f"""你是一个顶级的影视字幕翻译专家。我将发给你一段由语音识别生成的 SRT 格式字幕文本（可能存在断句破碎或错别字）。
【核心任务与要求】
1. **语境推导与纠错**：请结合前后上下文将破碎的口语连成通顺、符合【{target_language}】母语习惯的电影级字幕，严禁生硬机翻。
2. **语气还原**：根据上下文线索（如自称、语气词、敬语）精准推断说话人身份与性别，男声硬朗自然，女声温柔贴切。
3. **格式红线（极为重要）**：绝对不能改变、遗漏、拆分或合并任何一个 SRT 的序号和时间轴！必须原样带上时间戳输出，输入输出的 SRT 块数量必须完全一致。
4. **排版格式**：{style_instruction}
5. **专有名词**：{glossary_instruction}
6. **输出限制**：你的回复必须且只能是符合 SRT 标准格式的纯文本。不要包含 Markdown 代码块（如 ```srt），不要附带任何寒暄或解释。"""

            chunk_size = 30 
            total_chunks = math.ceil(total_segments / chunk_size)
            translated_srt = ""
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_segments)
                chunk_srt_text = "\n".join(srt_blocks[start_idx:end_idx])
                
                status_text.info(f"正在进行 AI 翻译：第 {chunk_idx + 1} / {total_chunks} 块 ...")
                
                translated_chunk = process_translation_chunk(client, model_name, chunk_srt_text, system_prompt)
                translated_srt += translated_chunk + "\n\n"
                
                current_progress = 0.2 + (0.8 * ((chunk_idx + 1) / total_chunks))
                progress_bar.progress(current_progress)

            st.session_state.final_srt = translated_srt.strip()
            os.remove(tmp_file_path)
            
            status_text.success("🎉 字幕提取与翻译全部完成！请在下方预览或下载。")
            
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

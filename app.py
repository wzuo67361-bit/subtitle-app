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
st.markdown("支持提取原字幕、多语言翻译、双语对照，以及智能语气还原。")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# 1. API 配置 (预设主流平台)
st.sidebar.subheader("1. API 设置 (翻译大脑)")

# 预设市面上主流且兼容 OpenAI 格式的平台
api_presets = {
    "DeepSeek (推荐, 极便宜)": {
        "url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "阿里通义千问 (Qwen)": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus"
    },
    "Kimi (月之暗面)": {
        "url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k"
    },
    "智谱 GLM": {
        "url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash"
    },
    "OpenAI 官方": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    },
    "自定义 (其他代理/中转站)": {
        "url": "",
        "model": ""
    }
}

selected_provider = st.sidebar.selectbox("选择大模型平台", list(api_presets.keys()))

# 根据选择自动填充网址和模型
if selected_provider == "自定义 (其他代理/中转站)":
    st.sidebar.info("💡 请查阅你所用平台的官方文档获取以下信息")
    base_url = st.sidebar.text_input("API 网址 (Base URL)", placeholder="例如: https://api.siliconflow.cn/v1")
    model_name = st.sidebar.text_input("模型名称 (Model)", placeholder="例如: Qwen/Qwen2.5-7B-Instruct")
else:
    base_url = api_presets[selected_provider]["url"]
    model_name = api_presets[selected_provider]["model"]
    st.sidebar.caption(f"🔗 自动配置网址: `{base_url}`")
    st.sidebar.caption(f"🤖 自动配置模型: `{model_name}`")

api_key = st.sidebar.text_input(f"输入 {selected_provider.split(' ')[0]} 的 API Key", type="password")

# 2. 语言与字幕选项
st.sidebar.subheader("2. 字幕设置")
source_lang = st.sidebar.selectbox(
    "视频源语言", 
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
st.sidebar.subheader("3. 专业词汇/专有名词校正")
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
    return WhisperModel("base", device="cpu", compute_type="int8")

def translate_srt_chunk(srt_chunk, target_opt, gloss, client, model):
    """调用 LLM 翻译 SRT 文本块，并加入智能角色推断"""
    system_prompt = f"""你是一个顶级的影视字幕翻译专家。请严格按照要求翻译以下 SRT 字幕。
目标要求：{target_opt}
专业词汇表：\n{gloss}

【核心翻译原则 - 智能角色与语气还原】：
1. 请根据原文（特别是日语中的自称如俺/僕/私，句尾语气词如わ/ぜ/ぞ，以及敬语/口语的使用）智能推断说话人的性别（男/女）、身份或情绪。
2. 翻译出的译文必须极其精准地符合该角色的语气！男性的台词要符合男性口吻，女性的台词要符合女性口吻。
3. 如果对话中明显有不同角色在交替说话，请在译文的自然表达中体现出差异，不要翻译成干巴巴的机器味。

【严格排版规则】：
1. 必须严格保留原有的 SRT 序号和时间轴（如 1 \n 00:00:01,000 --> 00:00:04,000）。
2. 如果要求双语，请将原文放在第一行，译文放在第二行。
3. 绝对不要输出任何 Markdown 标记（如 ```srt），直接输出纯文本。
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

st.write("### 📤 第一步：上传文件")
uploaded_file = st.file_uploader("支持 MP4, MP3, WAV, M4A 等格式", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("🚀 开始生成与翻译", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ 请先上传音视频文件！")
        st.stop()
    
    if "翻译" in target_option or "双语" in target_option:
        if not api_key or not base_url or not model_name:
            st.warning("⚠️ 请在左侧完整填写 API Key（如果是自定义平台，还需填写网址和模型）！")
            st.stop()

    # 1. 保存上传的文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # 进度显示
        st.write("### ⏳ 第二步：处理进度")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 2. 语音识别 (Whisper)
        status_text.info("🎧 正在提取语音并生成原字幕 (这可能需要几分钟，请耐心等待)...")
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
            status_text.info(f"🧠 正在调用 AI ({model_name}) 进行智能翻译与语气还原...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            # 将字幕分块（每 40 个字幕块请求一次，防止超出 LLM 上下文或被截断）
            chunk_size = 40
            translated_srt_pieces = []
            
            total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
            
            for i in range(total_chunks):
                chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                chunk_text = "\n".join(chunk_lines)
                
                status_text.info(f"🧠 正在翻译第 {i+1}/{total_chunks} 部分 (AI 正在分析角色语气)...")
                translated_chunk = translate_srt_chunk(chunk_text, target_option, glossary, client, model_name)
                
                # 如果报错，直接停止并显示错误
                if "翻译出错" in translated_chunk:
                    st.error(translated_chunk)
                    st.stop()
                    
                translated_srt_pieces.append(translated_chunk)
                
                # 更新进度条 (50% 到 100%)
                current_progress = 50 + int(50 * ((i + 1) / total_chunks))
                progress_bar.progress(current_progress)
                
            final_srt_text = "\n\n".join(translated_srt_pieces)
        else:
            progress_bar.progress(100)

        status_text.success("✅ 处理完成！请在下方预览并下载字幕。")

        # 4. 结果展示与下载
        st.write("---")
        st.write("### 👀 第三步：字幕预览与下载")
        
        # 超大预览框
        st.text_area("你可以直接在这里检查、修改生成的字幕内容：", final_srt_text, height=400)
        
        # 下载按钮
        st.download_button(
            label="⬇️ 确认无误，一键下载 .srt 字幕文件",
            data=final_srt_text,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ 处理过程中发生错误: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

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
st.markdown("极简操作：选模型 -> 填密钥 -> 上传视频 -> 自动出双语字幕。")

# --- 侧边栏配置区 ---
st.sidebar.header("⚙️ 选项配置")

# ==========================================
# 1. 傻瓜式模型选择 (底层自动路由，无需填网址)
# ==========================================
st.sidebar.subheader("1. 选择翻译大模型")

# 核心路由字典：将模型名称自动映射到对应的官方网址
MODEL_ROUTING_MAP = {
    # --- Google Gemini 家族 (2026最新版) ---
    "gemini-3.8-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-3.5-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-2.5-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-1.5-pro": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-1.5-flash": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini-pro": "https://generativelanguage.googleapis.com/v1beta/openai/",
    
    # --- 国内顶级大厂 ---
    "deepseek-chat (深度求索)": "https://api.deepseek.com/v1",
    "moonshot-v1-8k (Kimi)": "https://api.moonshot.cn/v1",
    "qwen-plus (阿里通义千问)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    
    # --- 硅基流动 (永久免费开源模型) ---
    "Qwen/Qwen2.5-7B-Instruct (硅基流动免费版)": "https://api.siliconflow.cn/v1"
}

# 让用户直接在下拉菜单选模型
selected_display_name = st.sidebar.selectbox(
    "请直接选择你要用的模型：", 
    list(MODEL_ROUTING_MAP.keys())
)

# 代码底层自动获取对应的网址和真实的模型名
auto_base_url = MODEL_ROUTING_MAP[selected_display_name]
# 清理显示名称，提取真实的模型ID传给服务器
actual_model_id = selected_display_name.split(" ")[0] 

# 唯一的输入框：API Key
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="请输入您的 Google AI Studio 密钥")

# ==========================================
# 2. 语言与字幕选项
# ==========================================
st.sidebar.subheader("2. 字幕设置")
source_lang = st.sidebar.selectbox("视频源语言", ["ja (日语)", "auto (自动识别)", "en (英语)", "zh (中文)"], index=0)

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

# ==========================================
# 3. 专业词汇校正
# ==========================================
st.sidebar.subheader("3. 专业词汇/专有名词校正")
glossary = st.sidebar.text_area(
    "输入翻译对照（如：人名、术语），每行一个",
    placeholder="例如：\n山田太郎 -> Yamada Taro\n术语A -> Term A",
    height=100
)

# --- 核心处理函数 ---

def format_timestamp(seconds: float):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

def call_llm_translator(prompt_text, user_content, key, url, model):
    """统一的大模型调用通道"""
    try:
        client = OpenAI(api_key=key.strip(), base_url=url.strip())
        response = client.chat.completions.create(
            model=model.strip(),
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        result = re.sub(r'^```(?:srt|text)?\n', '', result)
        return re.sub(r'\n```$', '', result)
    except Exception as e:
        return f"翻译出错: {str(e)}"

# --- 主界面执行逻辑 ---

st.write("### 📤 第一步：上传音视频文件")
uploaded_file = st.file_uploader("支持 MP4, MP3, WAV, M4A 等主流音视频格式", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("🚀 开始生成与翻译", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ 请先上传音视频文件！")
        st.stop()
    
    if "翻译" in target_option or "双语" in target_option:
        if not api_key:
            st.warning("⚠️ 请在左侧填入 API Key！")
            st.stop()
            
        # 严谨拦截绝对不可用的 AQ. 密钥，防止用户白白等待报错
        if api_key.strip().startswith("AQ."):
            st.error("🚨 密钥错误：检测到 `AQ.` 开头的谷歌企业云令牌！\n\n该令牌缺少项目编号，绝对无法在此网页使用。请去申请 `AIzaSy` 开头的谷歌官方密钥，或使用 `sk-` 开头的国内密钥！")
            st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        st.write("### ⏳ 第二步：处理进度")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. Faster-Whisper 音频提取
        status_text.info("🎧 正在使用 faster-whisper 提取原字幕 (请稍候)...")
        model = load_whisper_model()
        lang_code = source_lang.split(" ")[0]
        lang_param = None if lang_code == "auto" else lang_code
        
        segments, info = model.transcribe(tmp_file_path, language=lang_param, beam_size=5)
        
        original_srt_lines = []
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            original_srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
            
        original_srt_text = "\n".join(original_srt_lines)
        progress_bar.progress(50)

        # 2. AI 智能翻译与语气校正
        final_srt_text = original_srt_text
        
        if "翻译" in target_option or "双语" in target_option:
            status_text.info(f"🧠 正在调用大模型 ({actual_model_id}) 深入解析对话与语气...")
            
            system_prompt = f"""你是一个顶级的影视字幕翻译专家。目标任务：{target_option}。
专业词汇校对对照表：\n{glossary}

【核心翻译原则 - 智能角色与语气还原】：
1. 必须通盘理解日文上下文，根据自称（俺、僕、私、あたし等）、句尾终助词（わ、ぜ、ぞ、かしら等）以及敬语/简体的差异，精准推断说话人的性别与身份关系。
2. 翻译出的译文必须符合该角色的性格与语气！男性台词坚决展现男人口吻，女性台词体现女性口吻，坚决杜绝生硬机翻。
3. 严格保留原有的 SRT 序号和时间轴格式（如 1 \\n 00:00:01,000 --> 00:00:04,000）。
4. 若选择双语，第一行为原文，第二行为译文。
5. 绝对不要输出任何 Markdown 标记（如 ```srt），直接输出纯文本。"""

            chunk_size = 35
            translated_srt_pieces = []
            total_chunks = math.ceil(len(original_srt_lines) / chunk_size)
            
            for i in range(total_chunks):
                chunk_lines = original_srt_lines[i*chunk_size : (i+1)*chunk_size]
                chunk_text = "\n".join(chunk_lines)
                status_text.info(f"🧠 正在翻译第 {i+1}/{total_chunks} 组字幕 (角色语气分析中)...")
                
                # 底层自动使用映射好的网址和模型名进行请求
                translated_chunk = call_llm_translator(system_prompt, chunk_text, api_key, auto_base_url, actual_model_id)
                
                if "翻译出错" in translated_chunk:
                    st.error(translated_chunk)
                    st.stop()
                    
                translated_srt_pieces.append(translated_chunk)
                current_progress = 50 + int(50 * ((i + 1) / total_chunks))
                progress_bar.progress(current_progress)
                
            final_srt_text = "\n\n".join(translated_srt_pieces)
        else:
            progress_bar.progress(100)

        status_text.success("✅ 全部处理完成！请在下方预览并下载字幕。")

        # 3. 预览与下载
        st.write("---")
        st.write("### 👀 第三步：字幕预览与下载")
        st.text_area("字幕内容确认区（可直接在此处二次编辑）：", final_srt_text, height=400)
        
        st.download_button(
            label="⬇️ 一键下载 .srt 字幕文件",
            data=final_srt_text,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ 处理过程中发生异常: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

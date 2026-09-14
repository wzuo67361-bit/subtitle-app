import streamlit as st
import pandas as pd
import json
import io
import tempfile
import os
import re
import time
from PIL import Image
from google import genai
from google.genai import types

# ----------------- 页面基础配置 -----------------
st.set_page_config(
    page_title="AI 媒体翻译与数据工作站",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎛️ AI 媒体翻译与数据提取工作站")

# ----------------- 侧边栏配置 (API与模型选择) -----------------
with st.sidebar:
    st.header("⚙️ 核心设置")
    api_key = st.text_input("输入 Gemini API Key", type="password")
    
    # 官方推荐稳定模型 + 支持自定义扩展
    model_options = [
        "gemini-2.0-flash",        # 极速推荐，多模态能力强
        "gemini-1.5-flash",        # 高度稳定，抗过载能力强
        "gemini-1.5-pro",          # 强力模型，适合长难表格
        "gemini-2.0-flash-lite",   # 轻量化
        "自定义/其他模型"
    ]
    selected_option = st.selectbox("🤖 选择 AI 模型", model_options, index=0)
    
    if selected_option == "自定义/其他模型":
        selected_model = st.text_input("请输入具体的模型名称", value="gemini-2.0-flash")
    else:
        selected_model = selected_option
    
    if not api_key:
        st.warning("⚠️ 必须输入 API Key 才能唤醒系统功能。")
    st.markdown("---")
    st.markdown(f"**系统状态：**\n- 当前驱动模型：`{selected_model}`")

# ----------------- 初始化全局状态 -----------------
if "subtitle_df" not in st.session_state:
    st.session_state.subtitle_df = None
if "table_df" not in st.session_state:
    st.session_state.table_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- 增强版通用工具函数 -----------------
def generate_with_retry(client, model, contents, config, max_retries=3):
    """带自动重试的 API 调用函数，解决 503 UNAVAILABLE 问题"""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e)
            # 如果是 503 过载或 429 频控，且还有重试机会，则进行指数退避重试
            if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                st.toast(f"⏳ 官方服务器繁忙 (503)，正在进行第 {attempt + 1} 次重试 (等待 {wait_time} 秒)...", icon="⚠️")
                time.sleep(wait_time)
            else:
                raise e

def safe_extract_json(raw_text):
    """稳健的 JSON 解析器，防崩防截断"""
    if not raw_text or not raw_text.strip():
        raise ValueError("模型未返回任何文本内容（可能因图片过大、敏感词拦截或模型输出为空）。")
    
    text = raw_text.strip()
    # 剔除 Markdown 标记
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
        if text.startswith("json"):
            text = text[4:].strip()
            
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果直接解析失败，使用正则匹配最外层的 JSON 数组 [...]
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法将模型返回内容解析为标准表格 JSON。返回前200字符: {text[:200]}...")

# ----------------- 选项卡架构搭建 -----------------
tab1, tab2 = st.tabs(["🎵 视听字幕与翻译 (云端版)", "📊 图片表格提取与 AI 编辑器"])

# ==============================================================================
# TAB 1: 视听字幕与翻译
# ==============================================================================
with tab1:
    st.header("🎵 音视频智能字幕提取与双语翻译")
    media_file = st.file_uploader("上传音/视频文件", type=["mp3", "wav", "m4a", "mp4"])
    
    if media_file and api_key:
        if st.button("🚀 开始提取与翻译字幕", type="primary"):
            with st.spinner(f"🚀 正在使用 {selected_model} 处理媒体文件..."):
                try:
                    client = genai.Client(api_key=api_key)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media_file.name)[1]) as tmp_file:
                        tmp_file.write(media_file.read())
                        tmp_file_path = tmp_file.name

                    uploaded_media = client.files.upload(file=tmp_file_path)
                    
                    prompt = """
                    请仔细聆听此媒体文件，提取其中的所有语音，并翻译为中文。
                    必须以合法的 JSON 数组格式返回：[{"时间": "00:00-00:05", "原文": "Hello", "译文": "你好"}]。
                    绝不能包含任何 Markdown 符号或额外说明文字，只能输出纯 JSON 数组。
                    """
                    
                    response = generate_with_retry(
                        client=client,
                        model=selected_model,
                        contents=[uploaded_media, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192
                        )
                    )
                    
                    parsed_data = safe_extract_json(response.text)
                    st.session_state.subtitle_df = pd.DataFrame(parsed_data)
                    
                    client.files.delete(name=uploaded_media.name)
                    os.remove(tmp_file_path)
                    
                    st.success("✅ 字幕提取完成！")
                except Exception as e:
                    st.error(f"❌ 处理失败：{e}")

    if st.session_state.subtitle_df is not None:
        st.divider()
        edited_sub_df = st.data_editor(st.session_state.subtitle_df, use_container_width=True, key="sub_editor")
        st.session_state.subtitle_df = edited_sub_df
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            edited_sub_df.to_excel(writer, index=False)
        st.download_button("📥 下载字幕 Excel", data=buffer.getvalue(), file_name="字幕提取.xlsx")


# ==============================================================================
# TAB 2: 图片表格提取与 AI 编辑器
# ==============================================================================
with tab2:
    st.header("📊 严谨图片数据提取与 AI 代操助手")
    
    img_file = st.file_uploader("上传包含表格/数据的图片", type=["png", "jpg", "jpeg", "webp"])
    
    if img_file and api_key:
        col_img, col_btn = st.columns([1, 2])
        with col_img:
            st.image(Image.open(img_file), caption="原始图片", use_container_width=True)
            
        with col_btn:
            if st.button("🚀 开始精准提取表格", type="primary"):
                with st.spinner(f"正在使用 {selected_model} 解析表格，遇到服务器繁忙将自动重试..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        prompt = """
                        你现在的任务是极其严谨地识别图片中的表格数据。
                        
                        【硬性要求】：
                        1. 必须原封不动地提取每一行、每一列！绝对禁止遗漏任何一行数据！
                        2. 绝对禁止偷懒，禁止使用省略号(...)，必须从表格的第一行完整提取到最后一行！
                        3. 必须且仅输出标准的 JSON 数组（Array of Objects），每行一个 Object，Key 为列名，Value 为内容。
                        """
                        
                        response = generate_with_retry(
                            client=client,
                            model=selected_model,
                            contents=[types.Part.from_bytes(data=img_file.getvalue(), mime_type=img_file.type), prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.1,
                                max_output_tokens=8192
                            )
                        )
                        
                        parsed_data = safe_extract_json(response.text)
                        st.session_state.table_df = pd.DataFrame(parsed_data)
                        st.session_state.chat_history = [] 
                        st.success("✅ 完整提取成功！进入校对与智能编辑区。")
                    except Exception as e:
                        if "503" in str(e):
                            st.error("❌ 官方服务器当前极度拥堵 (503)。建议在侧边栏切换为 `gemini-1.5-flash` 模型后重试。")
                        else:
                            st.error(f"❌ 识别失败：{e}")

    # 编辑与对话交互区
    if st.session_state.table_df is not None:
        st.divider()
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("**1. 交互式数据表 (可双击修改)**")
            edited_table_df = st.data_editor(st.session_state.table_df, num_rows="dynamic", use_container_width=True, key="table_editor")
            st.session_state.table_df = edited_table_df

            buffer2 = io.BytesIO()
            with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
                edited_table_df.to_excel(writer, index=False)
            st.download_button("📥 下载数据 Excel", data=buffer2.getvalue(), file_name="表格数据.xlsx")
            
        with right_col:
            st.markdown(f"**🤖 AI 智能编辑助手 ({selected_model})**")
            chat_box = st.container(height=350)
            
            with chat_box:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        
            if user_cmd := st.chat_input("如：删除第一列，或者把单价乘以2"):
                st.session_state.chat_history.append({"role": "user", "content": user_cmd})
                with chat_box:
                    st.chat_message("user").markdown(user_cmd)
                    
                with st.spinner("AI 正在执行表格操作..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        sys_prompt = f"""
                        你是数据编辑助手。当前表格 JSON: {st.session_state.table_df.to_json(orient='records', force_ascii=False)}
                        用户指令: "{user_cmd}"
                        请输出 JSON 结构：{{"reply": "操作说明", "updated_json": [更新后的完整表格 JSON 数组，如无需更新则设为 null]}}
                        """
                        
                        res = generate_with_retry(
                            client=client,
                            model=selected_model,
                            contents=sys_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                max_output_tokens=8192
                            )
                        )
                        
                        ai_res = safe_extract_json(res.text)
                        
                        if isinstance(ai_res, dict) and ai_res.get("updated_json"):
                            st.session_state.table_df = pd.DataFrame(ai_res["updated_json"])
                            reply_text = f"{ai_res.get('reply', '执行完毕')} \n\n✅ **已更新左侧表格**"
                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                            st.rerun() 
                        else:
                            reply_text = ai_res.get("reply", "操作完成。") if isinstance(ai_res, dict) else "操作完成。"
                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                            st.chat_message("assistant").markdown(reply_text)
                            
                    except Exception as e:
                        st.error(f"助手处理出错：{e}")

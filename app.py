import streamlit as st
import pandas as pd
import json
import io
import tempfile
import os
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
    
    # 完整的模型下拉列表
    model_options = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-transcribe",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-pro"
    ]
    selected_model = st.selectbox("🤖 选择 AI 模型", model_options, index=7)
    
    if not api_key:
        st.warning("⚠️ 必须输入 API Key 才能唤醒系统功能。")
    st.markdown("---")
    st.markdown(f"**系统状态：**\n- 运行环境：轻量化云端架构\n- 当前模型：`{selected_model}`")

# ----------------- 初始化全局状态 (严格隔离) -----------------
if "subtitle_df" not in st.session_state:
    st.session_state.subtitle_df = None
if "table_df" not in st.session_state:
    st.session_state.table_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- 通用工具函数 -----------------
def clean_json_output(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
        if text.startswith("json"):
            text = text[4:].strip()
    return text

# ----------------- 选项卡架构搭建 -----------------
tab1, tab2 = st.tabs(["🎵 视听字幕与翻译 (云端版)", "📊 图片表格提取与 AI 编辑器"])

# ==============================================================================
# TAB 1: 视听字幕与翻译 (云端多模态解析)
# ==============================================================================
with tab1:
    st.header("🎵 音视频智能字幕提取与双语翻译")
    media_file = st.file_uploader("上传音/视频文件", type=["mp3", "wav", "m4a", "mp4"])
    
    if media_file and api_key:
        if st.button("🚀 开始提取与翻译字幕", type="primary"):
            with st.spinner(f"🚀 正在处理，请稍候..."):
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
                    
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[uploaded_media, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192 # 同样放开字幕的长度限制
                        )
                    )
                    
                    json_str = clean_json_output(response.text)
                    st.session_state.subtitle_df = pd.DataFrame(json.loads(json_str))
                    
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
# TAB 2: 图片表格提取与 AI 编辑器 (加入防偷懒机制)
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
                with st.spinner(f"正在使用 {selected_model} 逐行拆解表格，确保不遗漏..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # 【核心修正】：极其严厉的防偷懒提示词
                        prompt = """
                        你现在的任务是极其严谨地识别图片中的表格数据。
                        
                        【极度重要的硬性要求】：
                        1. 必须原封不动地提取每一行、每一列！绝对禁止遗漏任何一行数据！
                        2. 绝对禁止“偷懒”！禁止使用省略号(...)，禁止自作主张截断内容，必须从表格的第一行完整提取到最后一行！
                        3. 必须且仅输出标准的 JSON 数组（Array of Objects），每行一个 Object，Key 为列名，Value 为内容。
                        4. 不要 Markdown 标记，不要多余的废话。
                        """
                        
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[types.Part.from_bytes(data=img_file.getvalue(), mime_type=img_file.type), prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.1,         # 【核心修正】：降低温度，减少模型发散，提升精准度
                                max_output_tokens=8192   # 【核心修正】：最大化输出额度，防止长表格被强行截断
                            )
                        )
                        
                        json_str = clean_json_output(response.text)
                        st.session_state.table_df = pd.DataFrame(json.loads(json_str))
                        st.session_state.chat_history = [] 
                        st.success("✅ 完整提取成功！进入校对与智能编辑区。")
                    except Exception as e:
                        st.error(f"❌ 识别失败。可能图片过长或格式有误，错误详情：{e}")

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
                        
                        res = client.models.generate_content(
                            model=selected_model,
                            contents=sys_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                max_output_tokens=8192
                            )
                        )
                        
                        ai_res = json.loads(clean_json_output(res.text))
                        
                        if ai_res.get("updated_json"):
                            st.session_state.table_df = pd.DataFrame(ai_res["updated_json"])
                            reply_text = f"{ai_res.get('reply', '执行完毕')} \n\n✅ **已更新左侧表格**"
                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                            st.rerun() 
                        else:
                            reply_text = ai_res.get("reply", "操作完成。")
                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                            st.chat_message("assistant").markdown(reply_text)
                            
                    except Exception as e:
                        st.error(f"助手处理出错：{e}")

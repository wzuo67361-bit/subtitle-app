import streamlit as st
import pandas as pd
import json
from PIL import Image
from google import genai
from google.genai import types

# 页面基本配置
st.set_page_config(
    page_title="严谨图片数据与表格提取工作站",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 严谨图片数据与表格提取 (含 AI 编辑助手)")

# ----------------- 侧边栏配置 -----------------
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("Gemini API Key", type="password", help="请输入你的 Google Gemini API Key")
    if not api_key:
        st.warning("⚠️ 请先输入 API Key 以启动服务")

# 初始化 Session State
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- 核心功能函数 -----------------
def extract_table_from_image(image_bytes, mime_type, key):
    """利用 Gemini 2.0 提取图片中的表格并返回结构化 JSON"""
    client = genai.Client(api_key=key)
    prompt = """
    请严谨识别图片中的所有表格和数据。
    你必须且仅输出一个标准的 JSON 数组（Array of Objects），每个 Object 代表一行数据，Key 为列名，Value 为单元格内容。
    注意：
    1. 严禁包含 Markdown 格式标记（不要写 ```json ... ```）。
    2. 请确保数字和文本的准确性，不要漏掉任何行列。
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt
        ]
    )
    # 清理并解析 JSON
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()
            
    data = json.loads(raw_text)
    return pd.DataFrame(data)

def ask_ai_assistant(user_prompt, current_df, key):
    """AI 助手处理用户咨询与表格编辑请求"""
    client = genai.Client(api_key=key)
    
    system_instruction = f"""
    你是一个精通 Excel 和 Python Pandas 的高级数据编辑助手。
    当前用户正在处理一张数据表，数据的 JSON 格式如下：
    {current_df.to_json(orient='records', force_ascii=False)}

    用户的需求是："{user_prompt}"

    请严格按照以下规则响应：
    1. 如果用户只是提问（如“第二列是什么含义”、“总共有几行”），请直接给予专业、简洁的回答。
    2. 如果用户要求对表格进行编辑、修改、格式化或计算（如“删除第三行”、“把单价列乘以1.1”、“新增一列总价”）：
       你必须输出且仅输出可供 Python 评估执行的表达式或操作。为了确保安全和精准，你需要返回一个 JSON 对象，结构如下：
       {{
          "action": "explain_or_modify",
          "reply": "对用户操作的简要说明或解答",
          "updated_json": [全新的完整表格数据 JSON 数组，如果不需要修改表格则设为 null]
       }}
    请务必保证输出是合法的 JSON 格式！
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=system_instruction,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)


# ----------------- 主界面逻辑 -----------------
uploaded_file = st.file_uploader("上传包含表格或数据的图片", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file and api_key:
    col_img, col_action = st.columns([1, 2])
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="原始图片", use_container_width=True)
        
    with col_action:
        if st.button("🚀 开始严谨识别数据", type="primary"):
            with st.spinner("AI 正在高精度提取表格数据中..."):
                try:
                    bytes_data = uploaded_file.getvalue()
                    df = extract_table_from_image(bytes_data, uploaded_file.type, api_key)
                    st.session_state.df_data = df
                    st.session_state.chat_history = []  # 重置对话
                    st.success("数据提取成功！")
                except Exception as e:
                    st.error(f"识别失败，请检查 API Key 或图片清晰度。错误信息: {e}")

# 当已有识别好的表格时，进入编辑与 AI 交互区
if st.session_state.df_data is not None:
    st.divider()
    st.subheader("📋 数据预览与智能编辑工作站")
    
    # 左右双栏：左侧展示/编辑表格，右侧为 AI 助手
    left_col, right_col = st.columns([3, 2])
    
    # ---------- 左侧：动态可编辑表格与导出 ----------
    with left_col:
        st.markdown("**1. 交互式数据表（可直接双击单元格修改）**")
        # 使用 data_editor 实现用户直接在界面上编辑
        edited_df = st.data_editor(
            st.session_state.df_data,
            num_rows="dynamic",
            use_container_width=True,
            key="table_editor"
        )
        # 实时同步用户的直接手动修改
        st.session_state.df_data = edited_df

        # 导出区
        st.markdown("**2. 导出文件**")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            # 导出 Excel
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button(
                label="📥 下载 Excel 文件 (.xlsx)",
                data=buffer.getvalue(),
                file_name="extracted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with exp_col2:
            # 导出 Markdown
            md_text = edited_df.to_markdown(index=False)
            st.download_button(
                label="📝 下载 Markdown 文件 (.md)",
                data=md_text,
                file_name="extracted_data.md",
                mime="text/markdown"
            )

    # ---------- 右侧：AI 表格代操与咨询机器人 ----------
    with right_col:
        st.markdown("**🤖 AI 表格编辑助手**")
        st.caption("不会调格式？需要批量计算或删减？直接跟助手说！")
        
        # 对话历史渲染容器
        chat_container = st.container(height=350)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 输入框
        if prompt := st.chat_input("输入需求，如：把第一列删掉 / 帮我把单价列加上￥符号"):
            # 记录用户消息
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # 调用 AI 助手处理
            with st.spinner("AI 正在思考并执行操作..."):
                try:
                    res = ask_ai_assistant(prompt, st.session_state.df_data, api_key)
                    
                    # 判断 AI 是否修改了表格
                    if res.get("updated_json") is not None:
                        new_df = pd.DataFrame(res["updated_json"])
                        st.session_state.df_data = new_df
                        ai_reply = f"{res.get('reply', '修改已完成！')}\n\n✅ **已为你自动更新表格数据**"
                        # 强制刷新页面以实时展示左侧更新后的表格
                        st.rerun()
                    else:
                        ai_reply = res.get("reply", "操作完成。")

                    # 记录 AI 回复
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(ai_reply)

                except Exception as e:
                    st.error(f"助手处理请求时出现错误: {e}")

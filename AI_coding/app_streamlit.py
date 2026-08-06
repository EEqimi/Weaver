# app_streamlit.py
import json
from datetime import datetime

import streamlit as st

from agents.inspiration_catcher import InspirationCatcher
from agents.theme_deepener import ThemeDeepener
from utils.adapter import adapt_to_story_direction


# ============================================================
# 辅助函数
# ============================================================
def generate_creative_log(eval_history):
    """
    生成创作日志的辅助函数
    参数:
        eval_history: 存储了所有轮次评价结果的列表
    返回:
        格式化的 Markdown 字符串
    """
    title = "未命名故事"
    if "current_story" in st.session_state and st.session_state.current_story:
        title = st.session_state.current_story.get("title", "未命名故事")

    log = f"""# 《{title}》创作手记

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、作品评价历程

"""
    if not eval_history:
        log += "暂无评价记录。\n"
    else:
        for i, eval_data in enumerate(eval_history, 1):
            log += f"""
### 第 {i} 轮评价
- **总分**：{eval_data.get('overall_score', 0):.1f} / 10
- **各维度得分**：
"""
            for dim in eval_data.get('dimensions', []):
                log += f"  - {dim.get('name', '')}：{dim.get('score', 0)} 分\n"
            
            log += f"""
**整体评价**：{eval_data.get('summary', '暂无整体评价。')}

**主要修改建议**：
"""
            suggestions_count = 0
            for dim in eval_data.get('dimensions', []):
                suggestions = dim.get('suggestions', {})
                for macro in suggestions.get('macro', []):
                    log += f"- {macro}\n"
                    suggestions_count += 1
            if suggestions_count == 0:
                log += "- 暂无具体建议。\n"

    log += """
---

## 二、创作反思

（你可以根据上述评价建议，补充你在创作过程中的思考和收获）

- 
- 
- 

## 三、下一步计划

（你可以根据评价建议，规划下一步的修改方向或创作重点）

- 
- 
- 

---
*本创作手记由「文思工坊」自动生成，供你复盘与学习使用。*
"""
    return log


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="文思工坊 · 创作系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 自定义CSS样式 ----
st.markdown("""
<style>
    /* 全局字体微调 */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 主标题区样式 */
    .main-header {
        margin-bottom: 0.5rem;
        padding: 0.5rem 0 0.5rem 0;
        border-bottom: 2px solid #f0f2f6;
    }
    .main-header h1 {
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .main-header .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-top: -0.2rem;
    }
    .main-header .badge {
        background: #e9ecef;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #495057;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    /* 节点标题区 */
    .node-header {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .node-header h2 {
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        margin-bottom: 0.1rem !important;
    }
    .node-header .node-caption {
        color: #6c757d;
        font-size: 0.95rem;
    }
    
    /* 卡片式输入区域 */
    .input-card {
        background-color: #f8f9fa;
        padding: 1.8rem 2rem 1.2rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.8rem;
        border: 1px solid #e9ecef;
        transition: box-shadow 0.2s ease;
    }
    .input-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .input-card .card-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
        color: #212529;
    }
    .input-card .card-hint {
        font-size: 0.9rem;
        color: #868e96;
        margin-bottom: 1rem;
    }
    
    /* 按钮优化 */
    .stButton button {
        border-radius: 0.6rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12) !important;
    }
    .stButton button:active {
        transform: translateY(0px) !important;
    }
    
    /* 主按钮（primary）特殊样式 */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: white !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2d2d4a 0%, #1a2744 100%) !important;
    }
    
    /* 侧边栏优化 */
    .css-1d391kg, .css-1lcbmhc {
        padding-top: 1.5rem !important;
    }
    .sidebar-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    .sidebar-section:first-of-type {
        border-top: none;
        margin-top: 0;
        padding-top: 0;
    }
    
    /* 对话消息气泡优化 */
    .stChatMessage {
        border-radius: 0.75rem !important;
        margin-bottom: 0.8rem !important;
    }
    .stChatMessage .stChatMessageContent {
        padding: 0.8rem 1.2rem !important;
    }
    
    /* 分割线优化 */
    hr {
        margin: 1.5rem 0 !important;
        opacity: 0.6;
    }
    
    /* 成功/警告/信息消息优化 */
    .stAlert {
        border-radius: 0.6rem !important;
        border-left-width: 4px !important;
    }
    
    /* 展开器优化 */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        border-radius: 0.5rem !important;
    }
    
    /* 页脚 */
    .footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #f0f2f6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 精致标题区
# ============================================================
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 文思工坊</h1>
        <div class="subtitle">
            人机协作叙事生成系统
            <span class="badge">v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    st.caption("")
    st.caption("⚡ 七个创作节点")

# ============================================================
# 门户页模式（首次进入时显示）
# ============================================================

# 初始化门户状态
if "enter_workspace" not in st.session_state:
    st.session_state.enter_workspace = False

# 如果还没有进入工作台，显示门户页
if not st.session_state.enter_workspace:
    
    # 全屏居中布局
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # 品牌标题区（上边距大幅缩减）
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 0.2rem 0;">
            <div style="font-size: 3.2rem; line-height: 1.2;">🧠</div>
            <h1 style="font-size: 2.6rem; font-weight: 700; margin: 0.1rem 0 0 0; 
                       background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                文思工坊
            </h1>
            <p style="font-size: 0.95rem; color: #6c757d; margin: 0.1rem 0 0.8rem 0;">
                人机协作叙事生成系统 <span style="background: #e9ecef; padding: 0.05rem 0.5rem; border-radius: 12px; font-size: 0.6rem; color: #495057; -webkit-text-fill-color: #495057;">v2.0</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 七个节点概览卡片（带小标题）
        st.markdown("""
        <div style="margin: 0.2rem 0 1rem 0;">
            <p style="text-align: center; font-size: 0.75rem; font-weight: 500; color: #868e96; letter-spacing: 2px; margin-bottom: 0.6rem;">
                — 七个创作节点 —
            </p>
            <div style="display: flex; justify-content: center;">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; max-width: 600px; width: 100%;">
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">💡</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">灵感捕捉器</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">🎯</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">主题深化师</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">👥</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">人物工坊</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">📐</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">情节建筑师</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">✍️</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">章节作家</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">🎨</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">风格调色盘</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 0.5rem 0.2rem; border-radius: 0.6rem; text-align: center; border: 1px solid #e9ecef;">
                        <div style="font-size: 1.2rem;">📊</div>
                        <div style="font-size: 0.6rem; font-weight: 600; color: #212529; line-height: 1.2;">评价迭代器</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 进入按钮
        if st.button("🚀 进入工作台", use_container_width=True, type="primary"):
            st.session_state.enter_workspace = True
            st.rerun()
        
        # 底部版权信息
        st.markdown("""
        <div style="text-align: center; color: #adb5bd; font-size: 0.7rem; margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #f0f2f6;">
            七步创作闭环 · 让每个有故事的人成为真正的写作者
        </div>
        """, unsafe_allow_html=True)
    
    # 门户页显示完后，停止后续代码执行
    st.stop()

# ============================================================
# 初始化智能体
# ============================================================
if "catcher" not in st.session_state:
    st.session_state.catcher = InspirationCatcher()
if "deepener" not in st.session_state:
    st.session_state.deepener = ThemeDeepener()
if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 控制面板")
    
    node_option = st.selectbox(
        "选择创作节点",
        ["灵感捕捉器", "主题深化师", "人物工坊", "情节建筑师", "章节作家", "风格调色盘", "评价迭代器"]
    )
    
    st.divider()
    
    if st.button("🗑️ 清空所有数据", use_container_width=True):
        st.session_state.catcher.clear_history()
        st.session_state.deepener.clear_history()
        st.session_state.messages = []
        keys_to_clear = ["current_story", "theme_result", "character_result", "plot_result", "chapters", "style_result", "eval_history"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    st.markdown("### 💡 故事种子示例")
    example_seeds = [
        "一个在深夜便利店发生的爱情故事",
        "一个宇航员在火星上发现生命痕迹",
        "一个老人在临终前写给未来孙子的信",
    ]
    for seed in example_seeds:
        if st.button(seed, key=seed, use_container_width=True):
            st.session_state._input = seed
            st.rerun()
    
    st.divider()
    st.caption("文思工坊 · 七步创作闭环")


# ============================================================
# 节点一：灵感捕捉器
# ============================================================
if node_option == "灵感捕捉器":
    st.markdown("""
    <div class="node-header">
        <h2>💡 灵感捕捉器</h2>
        <div class="node-caption">输入你的故事种子，AI 将为你生成结构化的精彩方向</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 卡片式输入区域 ----
    with st.container():
        st.markdown("""
        <div class="input-card">
            <div class="card-title">✍️ 你的故事种子</div>
            <div class="card-hint">一句话描述你的灵感，剩下的交给AI</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            user_idea = st.text_input(
                "故事种子",
                placeholder="例如：一个在末日废墟中寻找最后一片绿叶的机器人。",
                key="user_input",
                value=st.session_state.get("_input", ""),
                label_visibility="collapsed"
            )
        with col2:
            tone = st.selectbox(
                "基调",
                ["不限", "悬念", "温馨", "史诗", "惊悚", "幽默"],
                key="tone",
                label_visibility="collapsed"
            )
        with col3:
            length = st.selectbox(
                "篇幅",
                ["不限", "短篇", "中篇", "长篇"],
                key="length",
                label_visibility="collapsed"
            )
        
        send_button = st.button("🚀 生成灵感", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 对话历史 ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- 处理用户输入 ----
    if send_button and user_idea:
        enhanced_prompt = f"用户想法：{user_idea}\n期望基调：{tone}\n期望篇幅：{length}"

        with st.chat_message("user"):
            st.markdown(f"**想法**：{user_idea}\n**基调**：{tone} | **篇幅**：{length}")
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"**想法**：{user_idea}\n**基调**：{tone} | **篇幅**：{length}",
            }
        )

        with st.chat_message("assistant"):
            with st.spinner("🧠 AI 正在构思故事方向..."):
                raw_result = st.session_state.catcher.run(enhanced_prompt)
                formatted_result = st.session_state.catcher.format_output(raw_result)

                try:
                    clean = raw_result.strip()
                    if clean.startswith("```json"):
                        clean = clean[7:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    story_data = json.loads(clean.strip())
                    if "directions" in story_data and story_data["directions"]:
                        st.session_state.current_story = story_data["directions"][0]
                except:
                    pass

                st.markdown(formatted_result)
        st.session_state.messages.append(
            {"role": "assistant", "content": formatted_result}
        )

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 欢迎来到文思工坊！

            在左侧输入你的故事种子（一句话想法），选择你期望的基调和篇幅，我就会为你生成 3-5 个结构化的故事方向。

            每个方向会包含：标题、独特设定、核心人物、主要冲突、可能主题、标志性意象和引子段落。

            试试看吧！✨
            """)


# ============================================================
# 节点二：主题深化师
# ============================================================
elif node_option == "主题深化师":
    st.markdown("""
    <div class="node-header">
        <h2>🎯 主题深化师</h2>
        <div class="node-caption">提炼故事的核心命题、情感基调与关键词体系</div>
    </div>
    """, unsafe_allow_html=True)

    source = st.radio(
        "故事来源", ["使用上一个节点的输出", "直接输入故事描述"], horizontal=True
    )

    story_data = None

    if source == "使用上一个节点的输出":
        if "current_story" in st.session_state and st.session_state.current_story:
            story_data = st.session_state.current_story
            st.success("✅ 已加载来自灵感捕捉器的故事方向")
            with st.expander("查看当前故事方向"):
                st.json(story_data)
        else:
            st.warning(
                "⚠️ 未找到上一个节点的输出，请先在灵感捕捉器中生成故事方向，或选择「直接输入故事描述」"
            )
    else:
        user_desc = st.text_area(
            "📖 请用3-5段话描述你的故事",
            placeholder="例如：我想写一个关于记忆的故事。主角是一位在记忆诊所工作的护士...",
            height=150
        )
        if st.button("📖 解析故事结构", use_container_width=True):
            if user_desc:
                with st.spinner("正在解析你的故事描述..."):
                    try:
                        story_data = adapt_to_story_direction(user_desc)
                        st.session_state.current_story = story_data
                        st.success("✅ 解析成功！请确认以下故事结构：")
                        with st.expander("查看解析结果", expanded=True):
                            st.json(story_data)
                    except Exception as e:
                        st.error(f"❌ 解析失败：{e}")
            else:
                st.warning("⚠️ 请先输入故事描述")

    if story_data or ("current_story" in st.session_state and st.session_state.current_story):
        if story_data is None:
            story_data = st.session_state.current_story

        if story_data:
            st.divider()
            st.markdown("### ✏️ 主题初探（可选）")
            initial_theme = st.text_area(
                "你对这个故事主题的初步想法",
                value=story_data.get("possible_theme", ""),
                placeholder="例如：关于记忆与身份、遗忘与背叛...",
                height=80
            )

            if st.button("🧠 深化主题", use_container_width=True, type="primary"):
                if initial_theme:
                    story_data["possible_theme"] = initial_theme

                with st.spinner("🧠 AI 正在深化主题..."):
                    story_json = json.dumps(story_data, ensure_ascii=False)
                    raw_result = st.session_state.deepener.run(story_json)
                    formatted_result = st.session_state.deepener.format_output(raw_result)

                    st.divider()
                    st.markdown("### ✨ 主题深化结果")
                    st.markdown(formatted_result)
                    st.session_state.theme_result = raw_result


# ============================================================
# 节点三：人物工坊
# ============================================================
elif node_option == "人物工坊":
    st.markdown("""
    <div class="node-header">
        <h2>👥 人物工坊</h2>
        <div class="node-caption">基于故事主题，生成结构化的角色群像和关系图谱</div>
    </div>
    """, unsafe_allow_html=True)

    if "character_workshop" not in st.session_state:
        from agents.character_workshop import CharacterWorkshop
        st.session_state.character_workshop = CharacterWorkshop()

    source = st.radio(
        "故事来源", ["使用上游节点数据", "直接输入故事描述"], horizontal=True
    )

    story_context = None
    if source == "使用上游节点数据":
        if "current_story" in st.session_state and "current_theme" in st.session_state:
            story_context = {
                "story_direction": st.session_state.current_story,
                "theme_output": st.session_state.current_theme,
                "length": "short",
            }
            st.success("✅ 已加载来自上游节点的数据")
            with st.expander("查看当前故事上下文"):
                st.json(story_context)
        else:
            st.warning(
                "⚠️ 未找到上游数据，请选择「直接输入故事描述」或先在灵感捕捉器和主题深化师中生成内容"
            )
    else:
        user_desc = st.text_area(
            "📖 请详细描述你的故事背景",
            placeholder="例如：我想写一个关于记忆的故事。主角是一位在记忆诊所工作的护士...",
            height=150
        )
        if st.button("📖 智能解析故事要素", use_container_width=True):
            if user_desc:
                with st.spinner("正在解析故事背景..."):
                    try:
                        from utils.adapter import adapt_to_character_workshop
                        story_context = adapt_to_character_workshop(user_desc)
                        st.session_state.current_story = story_context.get("story_direction")
                        st.session_state.current_theme = story_context.get("theme_output")
                        st.success("✅ 解析成功！")
                        with st.expander("查看解析结果", expanded=True):
                            st.json(story_context)
                    except Exception as e:
                        st.error(f"❌ 解析失败：{e}")
            else:
                st.warning("⚠️ 请先输入故事描述")

    if story_context:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**当前篇幅**：{story_context.get('length', '未指定')}")
        with col2:
            if st.button("🧠 生成角色群像", use_container_width=True, type="primary"):
                with st.spinner("👥 AI 正在设计角色..."):
                    input_data = {
                        "story_direction": story_context.get("story_direction", {}),
                        "theme_output": story_context.get("theme_output", {}),
                        "length": story_context.get("length", "short"),
                    }
                    input_json = json.dumps(input_data, ensure_ascii=False)
                    raw_result = st.session_state.character_workshop.run(input_json)
                    formatted_result = st.session_state.character_workshop.format_output(raw_result)
                    st.divider()
                    st.markdown("### ✨ 角色生成结果")
                    st.markdown(formatted_result)
                    st.session_state.character_result = raw_result


# ============================================================
# 节点四：情节建筑师
# ============================================================
elif node_option == "情节建筑师":
    st.markdown("""
    <div class="node-header">
        <h2>📐 情节建筑师</h2>
        <div class="node-caption">基于叙事模型，将人物与主题组织为结构化的场景级情节大纲</div>
    </div>
    """, unsafe_allow_html=True)

    if "plot_architect" not in st.session_state:
        from agents.plot_architect import PlotArchitect
        st.session_state.plot_architect = PlotArchitect()

    source = st.radio(
        "故事来源", ["使用上游节点数据", "直接输入故事前提"], horizontal=True
    )

    story_context = None
    if source == "使用上游节点数据":
        if all(key in st.session_state for key in ["current_story", "current_theme", "character_result"]):
            try:
                char_data = json.loads(st.session_state.character_result)
                story_context = {
                    "story_direction": st.session_state.current_story,
                    "theme_output": st.session_state.current_theme,
                    "character_output": char_data,
                    "length": "short",
                }
                st.success("✅ 已加载来自上游节点的完整数据")
                with st.expander("查看当前故事上下文"):
                    st.json(story_context)
            except:
                st.warning("⚠️ 人物数据解析失败，请检查上游节点输出")
        else:
            st.warning(
                "⚠️ 未找到完整上游数据，请先在灵感捕捉器、主题深化师和人物工坊中生成内容"
            )
    else:
        user_input = st.text_area(
            "📝 请用一句话或一段话概括你的故事核心",
            placeholder='例如："一个能看见他人记忆的统计学家，在调查数据异常时，发现了一个试图隐藏自我的AI。"',
            height=100
        )
        if st.button("📖 智能解析故事结构", use_container_width=True):
            if user_input:
                with st.spinner("正在解析故事结构..."):
                    try:
                        from utils.adapter import adapt_to_plot_architect
                        story_context = adapt_to_plot_architect(user_input)
                        st.session_state.current_story = story_context.get("story_direction")
                        st.session_state.current_theme = story_context.get("theme_output")
                        st.success("✅ 解析成功！")
                        with st.expander("查看解析结果", expanded=True):
                            st.json(story_context)
                    except Exception as e:
                        st.error(f"❌ 解析失败：{e}")
            else:
                st.warning("⚠️ 请先输入故事前提")

    if story_context:
        st.divider()
        st.markdown("### 📐 叙事模型选择")
        col1, col2 = st.columns([2, 1])
        with col1:
            length = story_context.get("length", "short")
            if length == "short":
                default_model = "起承转合"
                options = ["起承转合", "三幕式"]
            elif length == "medium":
                default_model = "三幕式"
                options = ["三幕式", "英雄之旅", "起承转合", "Save the Cat"]
            else:
                default_model = "英雄之旅"
                options = ["英雄之旅", "多线叙事", "三幕式"]
            selected_model = st.selectbox(
                "选择叙事模型（系统已根据篇幅推荐）",
                options,
                index=options.index(default_model) if default_model in options else 0,
            )
            st.caption(f"当前篇幅：{length}，推荐模型：{default_model}")
        with col2:
            chapter_count = st.number_input(
                "章节数",
                min_value=3,
                max_value=60,
                value=8 if length == "short" else (20 if length == "medium" else 40),
                step=1,
            )
        special_req = st.text_input(
            "特殊要求（可选）", placeholder="例如：我想要一个反转结局"
        )
        if st.button("🧠 生成情节大纲", use_container_width=True, type="primary"):
            with st.spinner("📐 AI 正在构建情节结构..."):
                input_data = {
                    "story_direction": story_context.get("story_direction", {}),
                    "theme_output": story_context.get("theme_output", {}),
                    "character_output": story_context.get("character_output", {}),
                    "user_choices": {
                        "selected_model": selected_model,
                        "custom_chapter_count": chapter_count,
                        "special_requirements": special_req if special_req else None,
                    },
                }
                input_json = json.dumps(input_data, ensure_ascii=False)
                raw_result = st.session_state.plot_architect.run(input_json)
                formatted_result = st.session_state.plot_architect.format_output(raw_result)
                st.divider()
                st.markdown("### ✨ 情节大纲生成结果")
                st.markdown(formatted_result)
                st.session_state.plot_result = raw_result


# ============================================================
# 节点五：章节作家
# ============================================================
elif node_option == "章节作家":
    st.markdown("""
    <div class="node-header">
        <h2>✍️ 章节作家</h2>
        <div class="node-caption">基于大纲与人物，逐章生成通顺、连贯的叙事文本</div>
    </div>
    """, unsafe_allow_html=True)

    if "chapter_writer" not in st.session_state:
        from agents.chapter_writer import ChapterWriter
        st.session_state.chapter_writer = ChapterWriter()
    
    if "chapters" not in st.session_state:
        st.session_state.chapters = []
    if "current_chapter_idx" not in st.session_state:
        st.session_state.current_chapter_idx = 0

    source = st.radio(
        "故事来源",
        ["使用上游节点数据", "独立启动（上传文件或填写描述）"],
        horizontal=True
    )

    story_context = None
    standalone_mode = False
    uploaded_content = None

    if source == "使用上游节点数据":
        required_keys = ["current_story", "current_theme", "character_result", "plot_result"]
        if all(key in st.session_state for key in required_keys):
            try:
                plot_data = json.loads(st.session_state.plot_result)
                story_context = {
                    "story_direction": st.session_state.current_story,
                    "theme_output": st.session_state.current_theme,
                    "character_output": json.loads(st.session_state.character_result),
                    "plot_output": plot_data,
                    "length": "short"
                }
                st.success("✅ 已加载来自上游节点的完整数据")
                with st.expander("查看当前故事上下文"):
                    st.json(story_context)
            except:
                st.warning("⚠️ 部分数据解析失败，请检查上游节点输出")
        else:
            st.warning("⚠️ 未找到完整上游数据，请先生成灵感、主题、人物和情节")
    else:
        standalone_mode = True
        st.markdown("### 📝 提供故事材料")
        input_method = st.radio("选择输入方式", ["填写文字描述", "上传文件"], horizontal=True)
        
        if input_method == "填写文字描述":
            user_text = st.text_area(
                "请粘贴你的故事梗概、人物设定或情节大纲",
                height=150
            )
            if st.button("📖 解析文字材料", use_container_width=True):
                if user_text:
                    with st.spinner("正在解析..."):
                        from utils.adapter import detect_continuation_mode, extract_existing_chapters
                        mode = detect_continuation_mode(user_text)
                        if mode == 'continuation':
                            chapters = extract_existing_chapters(user_text)
                            st.session_state.chapters = chapters
                            st.success(f"检测到续写模式，已提取 {len(chapters)} 章")
                            uploaded_content = user_text
                        else:
                            st.success("检测到素材模式，已准备好生成")
                            uploaded_content = user_text
                else:
                    st.warning("⚠️ 请先输入文字")
        else:
            uploaded_file = st.file_uploader("上传文件", type=['txt', 'md', 'docx'])
            if uploaded_file is not None:
                try:
                    from utils.adapter import parse_uploaded_file, detect_continuation_mode, extract_existing_chapters
                    file_content = parse_uploaded_file(uploaded_file, uploaded_file.name.split('.')[-1])
                    mode = detect_continuation_mode(file_content)
                    st.info(f"检测到：{'续写模式' if mode == 'continuation' else '素材模式'}")
                    if mode == 'continuation':
                        chapters = extract_existing_chapters(file_content)
                        st.session_state.chapters = chapters
                        st.success(f"已提取 {len(chapters)} 章已有内容")
                    uploaded_content = file_content
                except Exception as e:
                    st.error(f"文件解析失败: {e}")

    st.divider()
    st.markdown("### ✍️ 写作偏好设置（可选）")
    col1, col2 = st.columns(2)
    with col1:
        perspective = st.selectbox("叙事视角", ["第三人称有限", "第一人称", "第三人称全知"])
        tense = st.selectbox("时态", ["过去时", "现在时"])
    with col2:
        pacing = st.selectbox("节奏", ["中", "快", "慢"])
        dialogue = st.selectbox("对话密度", ["中", "高", "低"])
    special_notes = st.text_input("其他要求", placeholder="例如：多用短句、每章结尾留悬念")

    if st.button("📖 开始写作（生成下一章）", use_container_width=True, type="primary"):
        if not story_context and not uploaded_content:
            st.warning("⚠️ 请先提供故事材料")
        else:
            with st.spinner("✍️ AI 正在写作..."):
                if story_context:
                    context_data = story_context
                else:
                    context_data = {"user_material": uploaded_content}
                context_data["writing_preferences"] = {
                    "narrative_perspective": perspective,
                    "tense": tense,
                    "pacing": pacing,
                    "dialogue_density": dialogue,
                    "special_notes": special_notes
                }
                if st.session_state.chapters:
                    context_data["existing_chapters"] = st.session_state.chapters
                    context_data["next_chapter_number"] = len(st.session_state.chapters) + 1
                input_json = json.dumps(context_data, ensure_ascii=False)
                raw_result = st.session_state.chapter_writer.run(input_json)
                formatted_result = st.session_state.chapter_writer.format_output(raw_result)
                st.divider()
                st.markdown("### ✨ 生成的章节")
                st.markdown(formatted_result)
                try:
                    clean = raw_result.strip()
                    if clean.startswith("```json"):
                        clean = clean[7:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    chapter_data = json.loads(clean.strip())
                    st.session_state.chapters.append(chapter_data)
                except:
                    st.info("章节已生成，但解析保存失败。")


# ============================================================
# 节点六：风格调色盘
# ============================================================
elif node_option == "风格调色盘":
    st.markdown("""
    <div class="node-header">
        <h2>🎨 风格调色盘</h2>
        <div class="node-caption">对故事进行文学风格的精细化润色，支持单一或混合风格</div>
    </div>
    """, unsafe_allow_html=True)

    if "style_palette" not in st.session_state:
        from agents.style_palette import StylePalette
        st.session_state.style_palette = StylePalette()
    
    source = st.radio(
        "稿件来源",
        ["使用上游节点稿件", "上传或粘贴新文本"],
        horizontal=True
    )
    
    manuscript_text = None
    
    if source == "使用上游节点稿件":
        if "chapters" in st.session_state and st.session_state.chapters:
            full_text = ""
            for ch in st.session_state.chapters:
                if isinstance(ch, dict):
                    full_text += ch.get("content", "") + "\n\n"
            if full_text:
                manuscript_text = full_text
                st.success(f"✅ 已加载章节作家生成的稿件（共 {len(full_text)} 字）")
                with st.expander("查看当前稿件"):
                    st.text_area("稿件内容", full_text, height=200)
            else:
                st.warning("⚠️ 章节内容为空，请先生成章节")
        else:
            st.warning("⚠️ 未找到章节作家生成的稿件，请先生成章节或选择上传文本")
    else:
        input_method = st.radio("输入方式", ["粘贴文本", "上传文件"], horizontal=True)
        if input_method == "粘贴文本":
            manuscript_text = st.text_area("请粘贴需要润色的文本", height=200)
        else:
            uploaded_file = st.file_uploader("上传文件", type=['txt', 'md', 'docx'])
            if uploaded_file is not None:
                try:
                    from utils.adapter import parse_uploaded_file
                    manuscript_text = parse_uploaded_file(uploaded_file, uploaded_file.name.split('.')[-1])
                    st.success(f"文件解析成功，共 {len(manuscript_text)} 字")
                except Exception as e:
                    st.error(f"文件解析失败: {e}")
    
    if manuscript_text:
        st.divider()
        st.markdown("### 🎨 选择风格")
        
        default_writers = ["张爱玲", "王小波", "博尔赫斯", "村上春树", "雷蒙德·卡佛", "汪曾祺", "金庸"]
        selected_writers = st.multiselect("选择作家（可多选）", default_writers)
        
        if selected_writers:
            st.markdown("#### 调整风格比例（总和自动归一化为100%）")
            weights = {}
            total = 0
            for writer in selected_writers:
                val = st.slider(f"{writer} 比例", 0, 100, 100 // len(selected_writers), key=f"weight_{writer}")
                weights[writer] = val
                total += val
            if total > 0:
                st.info(f"当前比例总和：{total}%，系统将自动归一化为100%")
            else:
                st.warning("请至少为一个作家设置大于0的比例")
        
        style_tags = st.multiselect(
            "或选择风格标签（可选）",
            ["细腻心理描写", "冷峻短句", "奇幻隐喻", "幽默反讽", "苍凉意象", "简洁直接"]
        )
        
        with st.expander("⚙️ 高级模式（可选）"):
            col1, col2 = st.columns(2)
            with col1:
                vocabulary = st.selectbox("词汇偏好", ["现代", "古雅", "口语化"])
                sentence_length = st.selectbox("句子长度", ["混合", "短句", "长句"])
            with col2:
                rhetoric = st.selectbox("修辞密度", ["适中", "精简", "浓郁"])
                narrative_focus = st.selectbox("叙事重点", ["均衡", "心理描写", "对话", "动作"])
        
        if st.button("🎨 应用风格润色", use_container_width=True, type="primary"):
            if not selected_writers and not style_tags:
                st.warning("⚠️ 请至少选择一位作家或一个风格标签")
            else:
                with st.spinner("🎨 AI 正在进行风格润色..."):
                    style_components = []
                    if selected_writers and total > 0:
                        for writer in selected_writers:
                            if weights.get(writer, 0) > 0:
                                style_components.append({
                                    "writer": writer,
                                    "weight": round(weights[writer] / total * 100)
                                })
                    input_data = {
                        "text": manuscript_text,
                        "style_components": style_components,
                        "custom_tags": style_tags,
                        "advanced_settings": {
                            "vocabulary": vocabulary,
                            "sentence_length": sentence_length,
                            "rhetoric": rhetoric,
                            "narrative_focus": narrative_focus
                        }
                    }
                    input_json = json.dumps(input_data, ensure_ascii=False)
                    raw_result = st.session_state.style_palette.run(input_json)
                    formatted_result = st.session_state.style_palette.format_output(raw_result)
                    
                    st.divider()
                    st.markdown("### ✨ 润色结果")
                    
                    try:
                        clean = raw_result.strip()
                        if clean.startswith("```json"):
                            clean = clean[7:]
                        if clean.endswith("```"):
                            clean = clean[:-3]
                        data = json.loads(clean.strip())
                        styled_text = data.get("styled_text", "")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 📄 原稿")
                            st.text_area("", manuscript_text, height=300, key="original")
                        with col2:
                            st.markdown("#### 🎨 润色版")
                            st.text_area("", styled_text, height=300, key="styled")
                        
                        choice = st.radio("选择版本", ["采用润色版", "保留原稿", "手动编辑"], horizontal=True)
                        if choice == "手动编辑":
                            edited = st.text_area("编辑后的文本", styled_text if styled_text else manuscript_text, height=200)
                            if st.button("保存编辑"):
                                st.session_state.style_result = edited
                                st.success("已保存编辑版本")
                        elif choice == "采用润色版":
                            if st.button("保存润色版"):
                                st.session_state.style_result = styled_text
                                st.success("已保存润色版本")
                        else:
                            if st.button("保存原稿"):
                                st.session_state.style_result = manuscript_text
                                st.success("已保存原稿")
                    except Exception as e:
                        st.markdown(formatted_result)
                        st.info("润色完成，但段落对比展示解析失败。")


# ============================================================
# 节点七：评价迭代器
# ============================================================
elif node_option == "评价迭代器":
    st.markdown("""
    <div class="node-header">
        <h2>📊 评价迭代器</h2>
        <div class="node-caption">对故事进行多维度评价，提供修改建议，支持迭代优化并生成创作日志</div>
    </div>
    """, unsafe_allow_html=True)

    if "evaluation_iterator" not in st.session_state:
        from agents.evaluation_iterator import EvaluationIterator
        st.session_state.evaluation_iterator = EvaluationIterator()
    
    if "eval_history" not in st.session_state:
        st.session_state.eval_history = []
    
    source = st.radio(
        "稿件来源",
        ["使用上游节点稿件", "上传或粘贴新文本"],
        horizontal=True
    )
    
    manuscript_text = None
    if source == "使用上游节点稿件":
        if "style_result" in st.session_state and st.session_state.style_result:
            manuscript_text = st.session_state.style_result
            st.success("✅ 已加载风格调色盘生成的稿件")
            with st.expander("查看当前稿件"):
                st.text_area("稿件内容", manuscript_text, height=200)
        else:
            st.warning("⚠️ 未找到风格调色盘生成的稿件，请先生成或选择上传文本")
    else:
        input_method = st.radio("输入方式", ["粘贴文本", "上传文件"], horizontal=True)
        if input_method == "粘贴文本":
            manuscript_text = st.text_area("请粘贴需要评价的文本", height=200)
        else:
            uploaded_file = st.file_uploader("上传文件", type=['txt', 'md', 'docx'])
            if uploaded_file is not None:
                try:
                    from utils.adapter import parse_uploaded_file
                    manuscript_text = parse_uploaded_file(uploaded_file, uploaded_file.name.split('.')[-1])
                    st.success(f"文件解析成功，共 {len(manuscript_text)} 字")
                except Exception as e:
                    st.error(f"文件解析失败: {e}")
    
    if manuscript_text:
        st.divider()
        st.markdown("### ⚖️ 评价权重设置")
        st.caption("调整各维度的重要程度，系统将按权重计算加权总分。")
        
        default_weights = {"情节逻辑": 4, "人物塑造": 4, "语言质感": 3, "主题表达": 3, "节奏把控": 3, "情感共鸣": 3}
        weights = {}
        
        col1, col2 = st.columns(2)
        for idx, (dim, default) in enumerate(default_weights.items()):
            with col1 if idx < 3 else col2:
                weight = st.slider(dim, 1, 5, default, key=f"weight_{dim}")
                weights[dim] = weight
                st.caption(f"权重：{weight} ★" * (weight // 2))
        
        if st.button("🔄 恢复默认权重"):
            for dim, default in default_weights.items():
                weights[dim] = default
            st.rerun()
        
        if source != "使用上游节点稿件":
            st.info("💡 提示：如果文本语言偏平，建议先到「风格调色盘」进行润色后再来评价。")
        
        if st.button("📊 开始评价", use_container_width=True, type="primary"):
            with st.spinner("📊 AI 正在进行多维度评价..."):
                eval_context = {
                    "text": manuscript_text,
                    "user_weights": weights,
                    "iteration_round": len(st.session_state.eval_history) + 1
                }
                if st.session_state.eval_history:
                    eval_context["previous_evaluation"] = st.session_state.eval_history[-1]
                input_json = json.dumps(eval_context, ensure_ascii=False)
                raw_result = st.session_state.evaluation_iterator.run(input_json)
                formatted_result = st.session_state.evaluation_iterator.format_output(raw_result)
                st.divider()
                st.markdown("### ✨ 评价结果")
                st.markdown(formatted_result)
                try:
                    clean = raw_result.strip()
                    if clean.startswith("```json"):
                        clean = clean[7:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    eval_data = json.loads(clean.strip())
                    st.session_state.eval_history.append(eval_data)
                    st.success(f"✅ 第{len(st.session_state.eval_history)}轮评价已保存")
                except:
                    st.info("评价完成，但数据保存失败。")
    
    if st.session_state.eval_history:
        st.divider()
        st.markdown("### 📈 迭代历史")
        for i, eval_data in enumerate(st.session_state.eval_history, 1):
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                st.markdown(f"**第{i}轮**")
            with col2:
                st.markdown(f"总分：{eval_data.get('overall_score', 0):.1f}")
            with col3:
                if i > 1:
                    prev = st.session_state.eval_history[i-2].get('overall_score', 0)
                    curr = eval_data.get('overall_score', 0)
                    change = curr - prev
                    st.markdown(f"{'↑' if change > 0 else '↓'} {abs(change):.1f} 分")
        
        st.divider()
        if st.button("📝 生成创作日志", use_container_width=True):
            with st.spinner("正在生成创作日志..."):
                log = generate_creative_log(st.session_state.eval_history)
                st.download_button(
                    label="📥 下载创作日志 (Markdown)",
                    data=log,
                    file_name="创作手记.md",
                    mime="text/markdown"
                )


# ============================================================
# 页脚
# ============================================================
st.markdown("""
<div class="footer">
    文思工坊 · 人机协作叙事生成系统 · 七步创作闭环
</div>
""", unsafe_allow_html=True)
# utils/knowledge_base.py
"""知识库基类：作家风格库查阅（与 TextEditor 并联）

供各节点按需「临时调出」，供作者查阅作家风格，并可一键把风格摘要复制到文本编辑框。
"""
import json
import pathlib

import streamlit as st


class KnowledgeBase:
    """知识库基类：作家风格库查阅，与 TextEditor 并联"""

    WRITERS_DIR = pathlib.Path(__file__).resolve().parent.parent / "knowledge_base" / "data" / "writers"

    def __init__(self, editor):
        # editor 为并联的 TextEditor 实例，用于「复制到编辑框」
        self.editor = editor

    # ============ 数据加载 ============
    def load_writers(self):
        """遍历 writers/*.json，返回作家字典列表"""
        writers = []
        if not self.WRITERS_DIR.exists():
            return writers
        for path in sorted(self.WRITERS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("name"):
                    writers.append(data)
            except Exception:
                continue
        return writers

    def _style_summary(self, writer):
        """生成某作家的风格摘要文本（用于复制到编辑框）"""
        tags = "、".join(writer.get("core_style_tags", []))
        desc = writer.get("style_description", "")
        return f"【{writer.get('name', '')}】\n风格标签：{tags}\n风格描述：{desc}"

    # ============ 渲染 ============
    def render(self, compact=False):
        """渲染作家风格库：下拉选择 + 风格卡片 + 复制到编辑框"""
        writers = self.load_writers()
        st.markdown("### 📚 知识库（作家风格库）")
        if not writers:
            st.caption("知识库为空，请先在 knowledge_base/data/writers/ 放入作家 JSON 文件。")
            return

        # 复制反馈（来自上一次点击）
        fb_key = f"kb_copy_feedback::{self.editor.node_name}"
        feedback = st.session_state.pop(fb_key, None)
        if feedback:
            st.success(f"✅ 已将「{feedback}」的风格摘要复制到编辑框")

        # 作家选择
        names = [w.get("name", "") for w in writers]
        selected_name = st.selectbox(
            "选择作家查阅",
            names,
            key=f"kb_writer::{self.editor.node_name}",
            label_visibility="collapsed" if compact else "visible",
        )
        writer = next((w for w in writers if w.get("name") == selected_name), writers[0])

        # 风格卡片
        tags = writer.get("core_style_tags", [])
        desc = writer.get("style_description", "")
        works = writer.get("representative_works", [])
        if compact and len(desc) > 120:
            desc = desc[:120] + "…"

        st.markdown(f"**{writer.get('name', '')}**（{writer.get('era', '')} · {writer.get('region', '')}）")
        if tags:
            st.markdown("**风格标签**：" + "、".join(tags))
        if desc:
            st.markdown("**风格描述**：" + desc)
        if works and not compact:
            st.markdown("**代表作**：" + "、".join(works))

        # 复制到并联的编辑框
        if st.button(
            "📋 复制到编辑框",
            key=f"kb_copy::{self.editor.node_name}",
            use_container_width=True,
        ):
            self.editor.set_text(self._style_summary(writer))
            st.session_state[fb_key] = writer.get("name", "")
            st.rerun()

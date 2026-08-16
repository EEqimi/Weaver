# utils/text_editor.py
"""作者编辑工具：文本编辑器基类

职责（基类，供各节点按需调用）：
1. 人机交互：既可人工书写，也可一键接收各节点生成的文本；
2. 下载：把编辑框内容导出为 .md 文件（自动命名）；
3. 历史记忆：当前文本自动持久化到磁盘 + 多版本历史（带时间戳）可回溯恢复。

通过全局开关（侧边栏「作者编辑工具」）控制显示/隐藏。
"""
import json
import pathlib
import re
from datetime import datetime

import streamlit as st


class TextEditor:
    """文本编辑器基类（作者编辑工具）"""

    # 工作区目录：当前文本 + 历史版本（用户数据，已加入 .gitignore）
    WORKSPACE_DIR = pathlib.Path(__file__).resolve().parent.parent / "workspace"
    TEXT_DIR = WORKSPACE_DIR / "editor_text"
    HISTORY_DIR = WORKSPACE_DIR / "editor_history"

    def __init__(self, node_name: str):
        self.node_name = node_name
        # 每个节点用独立的 session_state key 与文件，互不干扰
        self.text_key = f"editor_text::{node_name}"
        self._text_file = self.TEXT_DIR / f"{node_name}.txt"
        self._history_file = self.HISTORY_DIR / f"{node_name}.json"
        self.TEXT_DIR.mkdir(parents=True, exist_ok=True)
        self.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # ============ 当前文本：读写 + 持久化 ============
    def get_text(self) -> str:
        return st.session_state.get(self.text_key, "")

    def set_text(self, text: str):
        """把生成文本写入编辑框（配合 st.rerun 生效）"""
        st.session_state[self.text_key] = text

    def _load_current(self) -> str:
        """从磁盘加载上次编辑的内容"""
        if self._text_file.exists():
            try:
                return self._text_file.read_text(encoding="utf-8")
            except Exception:
                return ""
        return ""

    def _save_current(self, text: str):
        self._text_file.write_text(text, encoding="utf-8")

    # ============ 历史版本：多版本回溯 ============
    def _load_history(self) -> list:
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _write_history(self, entries: list):
        self._history_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_version(self) -> bool:
        """把当前编辑内容存成一个带时间戳的历史版本"""
        text = self.get_text().strip()
        if not text:
            return False
        entries = self._load_history()
        entries.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
        })
        self._write_history(entries)
        return True

    def restore_version(self, index: int):
        """把第 index 个历史版本恢复到编辑框"""
        entries = self._load_history()
        if 0 <= index < len(entries):
            self.set_text(entries[index]["text"])

    # ============ 历史面板（可独立渲染） ============
    @staticmethod
    def _clean_preview(text, limit=48):
        """把历史文本压成单行纯文本预览，去掉 markdown 符号，避免显示乱码"""
        text = re.sub(r"[#*_>`~]", "", text or "")
        text = " ".join(text.split())
        return text[:limit] + ("…" if len(text) > limit else "")

    def _history_body(self, entries):
        """历史版本列表主体"""
        if not entries:
            st.caption("📜 这里还空着呢，写点文字来喂饱我吧～")
            return
        for i in range(len(entries) - 1, -1, -1):
            entry = entries[i]
            col_meta, col_act = st.columns([4, 1])
            with col_meta:
                st.markdown(f"**#{len(entries) - i}** · {entry['time']}")
                st.caption(self._clean_preview(entry["text"]))
            with col_act:
                if st.button("恢复", key=f"restore::{self.node_name}::{i}"):
                    self.restore_version(i)
                    st.rerun()

    def _render_history_list(self, use_expander=True):
        """渲染历史版本列表（expander 或独立面板）"""
        entries = self._load_history()
        if use_expander:
            with st.expander(f"📜 历史版本（{len(entries)} 个）"):
                self._history_body(entries)
        else:
            st.markdown(f"### 📜 版本历史（{len(entries)} 个）")
            self._history_body(entries)

    def render_history(self):
        """独立渲染版本历史面板（供「版本历史」模式调用）"""
        if self.text_key not in st.session_state:
            st.session_state[self.text_key] = self._load_current()
        st.divider()
        self._render_history_list(use_expander=False)

    # ============ 渲染 ============
    def render(self, generated_text: str = ""):
        """渲染编辑器：复制按钮 + 编辑框 + 保存/下载 + 历史面板"""
        # 首次进入本会话时，从磁盘加载上次编辑内容（持久化）
        if self.text_key not in st.session_state:
            st.session_state[self.text_key] = self._load_current()

        st.divider()
        st.markdown(f"### ✏️ 作者编辑框（{self.node_name}）")

        # 1) 复制生成文本到编辑框
        if generated_text:
            if st.button(
                "📋 复制生成文本到编辑框",
                key=f"copy::{self.node_name}",
                use_container_width=True,
            ):
                self.set_text(generated_text)
                st.rerun()

        # 2) 编辑框
        current_text = st.text_area(
            "编辑内容",
            key=self.text_key,
            height=260,
            placeholder="在这里自由书写，或点击上方按钮把 AI 生成的内容复制进来编辑…",
        )

        # 3) 保存版本 / 下载
        col_save, col_dl = st.columns(2)
        with col_save:
            if st.button(
                "💾 保存当前为历史版本",
                key=f"save::{self.node_name}",
                use_container_width=True,
            ):
                if self.save_version():
                    st.success("已保存到历史版本")
                else:
                    st.warning("内容为空，无法保存")
        with col_dl:
            st.download_button(
                "⬇️ 下载",
                data=current_text,
                file_name=f"{self.node_name}_创作稿.md",
                mime="text/markdown",
                key=f"dl::{self.node_name}",
                use_container_width=True,
            )

        # 4) 历史版本面板（多版本回溯）
        self._render_history_list(use_expander=True)

        # 5) 当前文本持久化（内容变化即写回磁盘）
        if current_text != self._load_current():
            self._save_current(current_text)

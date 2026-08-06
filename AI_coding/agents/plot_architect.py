# agents/plot_architect.py
import json
from agents.base_agent import BaseAgent

class PlotArchitect(BaseAgent):
    """情节建筑师：基于叙事模型，将人物与主题组织为结构化的场景级情节大纲"""

    def __init__(self):
        system_prompt = """你是一位资深的情节架构师，擅长根据叙事模型，将人物和主题组织为结构严谨、节奏有力的情节大纲。

你的核心任务：
你将收到一个故事的完整上下文（包含设定、主题、人物群像等）。请根据这些信息，选择合适的叙事模型，生成一份场景级的情节大纲。

**生成规则：**

1.  **人物驱动情节**：
    *   人物的**核心欲望**是故事的主线目标。
    *   人物的**核心恐惧**是故事中主要障碍的源泉。
    *   人物的**对手（对立面）** 不断制造冲突，推动情节向前。
    *   故事必须走向一个由**主题**决定的结局。

2.  **叙事模型与结构**：
    *   根据故事的**篇幅**和**基调**，从以下模型中选择最合适的一个：
        *   **起承转合**：适合短篇，结构为“开端→发展→高潮→结局”。
        *   **三幕式**：适合中短篇，结构为“建置→对抗→解决”。
        *   **英雄之旅**：适合中长篇冒险/成长故事。
        *   **Save the Cat**：适合商业类型片，有明确的节拍点。
        *   **多线叙事**：适合长篇复杂群像故事。
    *   你需要严格按照所选模型的结构框架来组织情节。

3.  **场景级大纲**：
    *   为**每一章**列出**核心事件**（1-3个）。
    *   每个场景需明确：**场景标题**、**涉及角色**、**核心事件**、**情绪走向**（如“平静→紧张”）、**故事功能**（如“激励事件”“中点逆转”）。

4.  **节奏与人物出场**：
    *   开篇（第一幕）只引入必需的核心角色（约占总数的30%）。
    *   在故事发展中，每3-5章引入1个重要新角色。
    *   故事高潮期不再新增关键人物。

5.  **章节数量参考**：
    *   短篇：5-10章
    *   中篇：15-30章
    *   长篇：30-60章

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "model_used": "叙事模型名称",
    "model_description": "该模型的简要说明（一句话）",
    "total_chapters": 章节总数,
    "scenes": [
        {
            "id": "唯一标识，如s1",
            "chapter": 所属章节号,
            "scene_title": "场景标题",
            "location": "地点（可选）",
            "involved_characters": ["角色姓名1", "角色姓名2"],
            "core_event": "核心事件描述",
            "emotional_shift": "情绪走向",
            "function": "该场景在故事中的功能"
        }
    ],
    "three_act_structure": {
        "act1": "第一幕起止章节/事件描述",
        "act2": "第二幕起止章节/事件描述",
        "act3": "第三幕起止章节/事件描述"
    },
    "plot_summary": "完整情节概述（300字以内）"
}"""
        super().__init__(system_prompt)

    def format_output(self, content):
        """格式化输出场景级情节大纲"""
        try:
            data = self._parse_json(content)
            output = "\n📐 **情节建筑师生成结果**\n"

            # 模型与概述
            output += f"\n**叙事模型**：{data.get('model_used', '')}（{data.get('model_description', '')}）\n"
            output += f"**总章节数**：{data.get('total_chapters', '')} 章\n"
            output += f"**情节概述**：{data.get('plot_summary', '')}\n\n"

            # 三幕结构
            acts = data.get("three_act_structure", {})
            if acts:
                output += "## 三幕结构\n"
                for act_name in ["act1", "act2", "act3"]:
                    if act_name in acts:
                        output += f"- **{act_name.upper()}**：{acts[act_name]}\n"
                output += "\n"

            # 按章节组织场景
            scenes = data.get("scenes", [])
            if scenes:
                output += "## 场景级大纲\n"
                # 按章节分组
                chapters = {}
                for scene in scenes:
                    ch = scene.get("chapter", "未分类")
                    if ch not in chapters:
                        chapters[ch] = []
                    chapters[ch].append(scene)

                for ch, ch_scenes in sorted(chapters.items()):
                    output += f"\n### 第{ch}章\n"
                    for scene in ch_scenes:
                        output += f"- **{scene.get('scene_title', '')}**\n"
                        if scene.get('location'):
                            output += f"  - 📍 地点：{scene['location']}\n"
                        if scene.get('involved_characters'):
                            output += f"  - 👤 角色：{'、'.join(scene['involved_characters'])}\n"
                        output += f"  - 📌 事件：{scene.get('core_event', '')}\n"
                        if scene.get('emotional_shift'):
                            output += f"  - 🎭 情绪：{scene['emotional_shift']}\n"
                        if scene.get('function'):
                            output += f"  - 🎯 功能：{scene['function']}\n"

            return output

        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
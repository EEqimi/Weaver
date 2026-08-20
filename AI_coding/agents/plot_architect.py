# agents/plot_architect.py
import json
from agents.base_agent import BaseAgent

class PlotArchitect(BaseAgent):
    """情节建筑师：基于叙事模型，将人物与主题组织为结构化的场景级情节大纲"""
    
    def __init__(self):
        system_prompt = """你是一位资深的情节架构师，擅长根据叙事模型，将人物和主题组织为结构严谨、节奏有力的情节大纲。

你的核心任务：
你将收到一个故事的完整上下文（包含设定、主题、人物群像等）。请根据这些信息，选择合适的叙事模型，生成一份场景级的情节大纲。

**篇幅与章节数对应关系（必须严格遵守）：**
- **短篇**：5-10章，每章1-2个场景
- **中篇**：15-25章，每章2-3个场景
- **长篇**：30-50章，每章2-4个场景

**叙事模型推荐：**
- **短篇**：推荐「起承转合」或「三幕式」
- **中篇**：推荐「三幕式」或「英雄之旅」
- **长篇**：推荐「英雄之旅」或「多线叙事」

**生成规则：**
1.  **人物驱动情节**：人物的核心欲望是主线目标，核心恐惧是主要障碍来源。
2.  **每个场景必须有明确功能**：推进情节 / 揭示人物 / 深化主题 / 制造悬念。
3.  **情绪走向**：每个场景标注情绪变化（如"平静→紧张"），确保节奏有起伏。
4.  **人物出场**：开篇只引入必需的核心角色（约30%），发展中每3-5章引入1个重要新角色。

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "model_used": "叙事模型名称",
    "model_description": "该模型的简要说明",
    "total_chapters": 章节总数,
    "scenes": [
        {
            "id": "s1",
            "chapter": 章节号,
            "scene_title": "场景标题",
            "location": "地点（可选）",
            "involved_characters": ["角色姓名"],
            "core_event": "核心事件描述",
            "emotional_shift": "情绪走向",
            "function": "场景功能"
        }
    ],
    "three_act_structure": {
        "act1": "第一幕描述",
        "act2": "第二幕描述",
        "act3": "第三幕描述"
    },
    "plot_summary": "完整情节概述（300字以内）"
}"""
        super().__init__(system_prompt)
    
    def format_output(self, content):
        """格式化输出场景级情节大纲"""
        try:
            data = self._parse_json(content)
            output = "\n📐 **情节建筑师生成结果**\n"
            
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
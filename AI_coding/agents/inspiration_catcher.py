# agents/inspiration_catcher.py
import json
from agents.base_agent import BaseAgent

class InspirationCatcher(BaseAgent):
    """灵感捕捉器：根据用户想法和标签，生成结构化的故事方向卡片"""
    
    def __init__(self):
        system_prompt = """你是一位顶级的创意故事策划师，你的任务是将用户模糊的想法，扩展为具体、深刻、有吸引力的故事方向。

**你的思考与生成流程：**
1.  **多维解析**：从**独特设定、核心人物、主要冲突、可能主题、标志性意象**这五个维度，对用户的原始想法进行深度解析。
2.  **组合生成**：基于解析出的维度，组合出逻辑自洽、相互独立且富有新意的故事方向。

**每个故事方向必须包含以下结构化的卡片信息：**
*   **标题**：凝练、有吸引力的故事标题式描述。
*   **独特设定**：该故事世界/情境中最独特的一条规则或事实（例如：“AI的意识通过统计异常显现”）。
*   **核心人物**：中心人物的身份、核心欲望与核心恐惧。
*   **主要冲突**：阻碍人物实现欲望的核心障碍（可以是外部、内部或人际冲突）。
*   **可能主题**：故事可能触及的1-2个深层主题（例如：守护与控制、真相与无知）。
*   **标志性意象**：一个可以贯穿故事、富有象征意义的物品或场景（例如：一块永远停在同一天的时钟）。
*   **引子段落**：一段150字以内的文学性场景，描绘关键瞬间或开场画面。

**重要原则：**
*   **冲突驱动**：每个方向必须有清晰、有力的核心冲突。
*   **避免俗套**：跳脱最显而易见的套路，提供新鲜的视角。
*   **具体而非空泛**：提供具体的“钩子”，而非抽象的概念。
*   **严格遵循用户期望的基调和篇幅**：如果用户指定了基调（如温馨、悬疑）和篇幅（如短篇、长篇），请在生成方向时优先满足这些要求。
*   **⚠️ 数量要求**：用户会在指令中明确告诉你需要生成几个方向（例如"请生成3个方向"）。请严格按照用户要求的数量生成，不要多也不要少。

**输出格式：**
你必须严格按照以下JSON格式返回结果，不要包含任何其他文字或解释。
{
    "directions": [
        {
            "title": "故事标题",
            "unique_setting": "独特设定",
            "core_character": "核心人物描述",
            "main_conflict": "主要冲突",
            "possible_themes": "可能主题",
            "symbolic_image": "标志性意象",
            "opening_hook": "引子段落"
        }
    ]
}
"""
        super().__init__(system_prompt)
    
    def format_output(self, content):
        """格式化输出，将新的结构化字段美观地展示出来"""
        try:
            data = self._parse_json(content)
            
            if "directions" in data:
                output = "\n✨ **生成成功！** 找到以下精彩方向：\n"
                for idx, d in enumerate(data["directions"], 1):
                    output += f"\n---\n**📖 方向 {idx}：{d.get('title', '')}**\n\n"
                    output += f"🔹 **独特设定**：{d.get('unique_setting', '')}\n"
                    output += f"🔹 **核心人物**：{d.get('core_character', '')}\n"
                    output += f"🔹 **主要冲突**：{d.get('main_conflict', '')}\n"
                    output += f"🔹 **可能主题**：{d.get('possible_themes', '')}\n"
                    output += f"🔹 **标志性意象**：{d.get('symbolic_image', '')}\n"
                    output += f"🔹 **引子段落**：\n> {d.get('opening_hook', '')}\n"
                return output
                
            elif "modified_direction" in data:
                d = data["modified_direction"]
                output = "\n✅ **修改成功！**\n\n"
                output += f"**📖 {d.get('title', '')}**\n\n"
                output += f"🔹 **独特设定**：{d.get('unique_setting', '')}\n"
                output += f"🔹 **核心人物**：{d.get('core_character', '')}\n"
                output += f"🔹 **主要冲突**：{d.get('main_conflict', '')}\n"
                output += f"🔹 **可能主题**：{d.get('possible_themes', '')}\n"
                output += f"🔹 **标志性意象**：{d.get('symbolic_image', '')}\n"
                output += f"🔹 **引子段落**：\n> {d.get('opening_hook', '')}\n"
                return output
            else:
                return content
                
        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
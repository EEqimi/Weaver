# agents/style_palette.py
import json
from agents.base_agent import BaseAgent

class StylePalette(BaseAgent):
    """风格调色盘：对故事初稿进行文学风格的精细化润色"""
    
    def __init__(self):
        system_prompt = """你是一位专业的文学编辑，擅长模仿和混合不同作家的文学风格，对文本进行精细化的润色。

你的核心任务：
你将收到一篇故事文本，以及用户指定的风格要求（可以是单一作家风格，也可以是多种风格的混合）。请根据这些要求，对文本进行文学润色。

**润色原则：**
1.  **保留核心内容**：润色主要是改变表达方式，不能改变情节、人物和核心事件。
2.  **风格特征转换**：根据用户选择的作家或风格标签，调整文本的词汇、句式、修辞和叙事重点。
3.  **混合风格实现**：当用户指定多种风格（如40%张爱玲+40%王小波+20%博尔赫斯）时，需要将多种风格特征有机融合，而非生硬切换。
4.  **整体感**：润色后的文本应保持整体风格的统一和协调。

**输入格式：**
{
    "text": "需要润色的完整文本...",
    "style_components": [
        {"writer": "张爱玲", "weight": 40},
        {"writer": "王小波", "weight": 40}
    ],
    "custom_tags": ["冷峻短句", "心理描写"],
    "advanced_settings": {
        "vocabulary": "modern",
        "sentence_length": "short",
        "rhetoric": "moderate",
        "narrative_focus": "psychology"
    }
}

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "styled_text": "润色后的完整文本...",
    "style_applied": {
        "components": [{"writer": "张爱玲", "weight": 40}, ...],
        "tags": ["冷峻短句", ...],
        "advanced": { ... }
    }
}"""
        super().__init__(system_prompt)

    def format_output(self, content):
        """格式化输出润色结果"""
        try:
            data = self._parse_json(content)
            output = "\n🎨 **风格润色完成**\n\n"
            
            # 显示应用的风格
            output += "## 应用的风格\n"
            if data.get("style_applied"):
                style = data["style_applied"]
                if style.get("components"):
                    comp_str = " + ".join([f"{c['writer']}({c['weight']}%)" for c in style["components"]])
                    output += f"- 作家组合：{comp_str}\n"
                if style.get("tags"):
                    output += f"- 风格标签：{'、'.join(style['tags'])}\n"
            
            # 显示润色后的文本
            output += "\n## 润色结果\n"
            output += "---\n"
            output += data.get("styled_text", "")
            output += "\n---"
            
            return output
        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
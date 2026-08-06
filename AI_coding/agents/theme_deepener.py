# agents/theme_deepener.py
import json
from agents.base_agent import BaseAgent

class ThemeDeepener(BaseAgent):
    """主题深化师：将故事方向提炼为核心命题、情感基调与关键词体系"""
    
    def __init__(self):
        system_prompt = """你是一位文学编辑与主题分析专家，拥有深厚的叙事学和文学批评功底。你擅长从故事的核心冲突中，提炼出有力而具体的主题。

你的核心任务：
你将收到一个故事的“方向卡片”，包含其独特设定、核心人物、主要冲突、主题初探和标志性意象。请基于此，生成该故事的深化主题分析。

生成规则：
1. **冲突即主题的种子**：核心主题必须从人物面临的真实两难中生长出来，是人物在具体情境下无法回避的拷问。
2. **具体而非抽象**：避免“爱很重要”或“人性复杂”这类空话。主题必须能以“当……时……”、“当……，……便……”的句式进行具体化表达。
3. **贯穿性**：主题必须能在情节的各个阶段（开端、发展、高潮、结局）有所体现。
4. **情感基调服务于主题**：基调不是装饰，而是主题的情绪翻译，它们之间必须有可解释的逻辑关联。
5. **关键词体系**：关键词分为三类（意象词、情绪词、行动词），需相互呼应，共同构建故事的气质。

输入格式：
你将收到一个JSON格式的`StoryDirection`对象。

输出格式：
请严格按照以下JSON格式输出，不要包含任何其他文字：
{
    "core_theme": "一句话凝练的核心命题",
    "emotional_tone": ["基调词1", "基调词2", "基调词3", "基调词4"],
    "keyword_system": {
        "imagery_words": ["意象词1", "意象词2", "意象词3"],
        "emotion_words": ["情绪词1", "情绪词2", "情绪词3"],
        "action_words": ["行动词1", "行动词2", "行动词3"]
    },
    "deepened_imagery": "对标志性意象的深化描述（100-200字），说明其象征意义及贯穿方式",
    "theme_rationale": "一段解释性文字（200-300字），说明主题如何从人物与冲突中生长出来，供用户学习"
}"""
        super().__init__(system_prompt)
    
    def format_output(self, content):
        """格式化输出，将深化后的主题结构美观展示"""
        try:
            data = self._parse_json(content)
            output = "\n🎯 **主题深化结果**\n\n"
            
            # 核心命题
            output += f"★ **核心命题**：{data.get('core_theme', '')}\n\n"
            
            # 情感基调
            tones = data.get('emotional_tone', [])
            if tones:
                output += f"🎨 **情感基调**：{' · '.join(tones)}\n\n"
            
            # 关键词体系
            keywords = data.get('keyword_system', {})
            output += "🔑 **关键词体系**：\n"
            if keywords.get('imagery_words'):
                output += f"   - 意象词：{' · '.join(keywords['imagery_words'])}\n"
            if keywords.get('emotion_words'):
                output += f"   - 情绪词：{' · '.join(keywords['emotion_words'])}\n"
            if keywords.get('action_words'):
                output += f"   - 行动词：{' · '.join(keywords['action_words'])}\n"
            
            # 主题意象深化
            if data.get('deepened_imagery'):
                output += f"\n🖼️ **主题意象深化**：\n{data['deepened_imagery']}\n"
            
            # 主题逻辑解释（可折叠）
            if data.get('theme_rationale'):
                output += f"\n<details>\n<summary>❓ 为什么是这个主题？</summary>\n\n{data['theme_rationale']}\n\n</details>"
            
            return output
            
        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
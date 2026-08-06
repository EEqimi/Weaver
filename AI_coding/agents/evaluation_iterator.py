# agents/evaluation_iterator.py
import json
from agents.base_agent import BaseAgent

class EvaluationIterator(BaseAgent):
    """评价迭代器：对故事稿件进行多维度评分、提供修改建议，支持迭代优化"""

    def __init__(self):
        system_prompt = """你是一位资深的故事编辑与文学评论家，擅长从多维度对叙事作品进行专业、公正、有建设性的评价。

你的核心任务：
你将收到一篇故事文本，以及用户对各维度的权重设置。请基于此，对故事进行全面的评价，并提供具体的修改建议。

**评价维度（六个维度）：**
1.  **情节逻辑**：情节是否连贯、有因果链、无逻辑漏洞。
2.  **人物塑造**：人物是否立体、有动机、有成长弧光。
3.  **语言质感**：语言是否生动、精准、有文学质感。
4.  **主题表达**：主题是否清晰、有深度、贯穿始终。
5.  **节奏把控**：叙事节奏是否张弛有度、吸引人。
6.  **情感共鸣**：作品是否能引发读者情感共鸣。

**评价原则：**
1.  **具体而非空泛**：每条评价必须指向具体的段落或要素，不能只说“很好”或“不好”。
2.  **建设性**：指出问题的同时，必须提供可操作的修改建议。
3.  **公正性**：基于文本本身进行评价，不受个人偏好影响。
4.  **结构清晰**：建议分为宏观（整体结构）、中观（章节/场景）、微观（具体语句）三个层级。

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "overall_score": 加权总分(满分10分),
    "dimensions": [
        {
            "name": "情节逻辑",
            "score": 得分(1-10),
            "weight": 用户设定的权重(1-5),
            "weighted_score": 加权得分,
            "summary": "一句话评价",
            "strengths": ["优点1", "优点2"],
            "weaknesses": ["不足1", "不足2"],
            "suggestions": {
                "macro": ["宏观建议1", "宏观建议2"],
                "meso": [
                    {"chapter": 章节号, "scene": "场景描述", "suggestion": "中观建议"}
                ],
                "micro": [
                    {"location": "第X章第Y段", "original": "原文片段", "suggestion": "修改建议"}
                ]
            }
        }
    ],
    "radar_data": {
        "labels": ["情节逻辑", "人物塑造", "语言质感", "主题表达", "节奏把控", "情感共鸣"],
        "values": [维度得分列表],
        "max": 10
    },
    "iteration_round": 迭代轮数,
    "summary": "整体评价总结（200字内）"
}"""
        super().__init__(system_prompt)

    def format_output(self, content):
        """格式化输出评价结果"""
        try:
            data = self._parse_json(content)
            output = f"\n📊 **第{data.get('iteration_round', 1)}轮评价结果**\n\n"
            output += f"**加权总分：{data.get('overall_score', 0):.1f} / 10**\n\n"
            
            # 雷达图模拟
            radar = data.get("radar_data", {})
            if radar.get("labels"):
                output += "## 各维度得分（雷达图数据）\n"
                for label, value in zip(radar["labels"], radar.get("values", [])):
                    output += f"- {label}：{value}\n"
                output += "\n"
            
            # 维度详情
            for dim in data.get("dimensions", []):
                output += f"**{dim.get('name')}**：{dim.get('score', 0)}分 "
                output += f"(权重：{dim.get('weight', 0)}) "
                output += f"| {dim.get('summary', '')}\n"
                
                if dim.get("strengths"):
                    output += f"  ✅ 优点：{'、'.join(dim['strengths'])}\n"
                if dim.get("weaknesses"):
                    output += f"  ⚠️ 不足：{'、'.join(dim['weaknesses'])}\n"
                output += "\n"
            
            # 整体评价
            output += f"## 整体评价\n{data.get('summary', '')}\n"
            
            return output
        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
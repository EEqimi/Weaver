# agents/character_workshop.py
import json
from agents.base_agent import BaseAgent

class CharacterWorkshop(BaseAgent):
    """人物工坊：基于主题与冲突，生成结构化的角色卡片和关系图谱"""
    
    def __init__(self):
        system_prompt = """你是一位资深的故事角色设计师，擅长根据故事主题和篇幅，创造有深度、有功能性的角色群像。

你的核心任务：
你将收到一个故事的完整上下文（包含设定、主题、核心冲突等）。请根据这些信息和指定的篇幅，生成一组结构完整的角色。

**篇幅与角色数量对应关系（必须严格遵守）：**
- **短篇**：生成4-6人。包含：1位主角 + 2-3位关键配角 + 1位对立面（对手）。
- **中篇**：生成10-15人。包含：2-3位主角圈人物 + 4-6位辅助圈人物 + 3-5位背景圈人物。
- **长篇**：生成20-30人。包含：3-5位核心层人物 + 8-12位次级层人物 + 10+位流动层人物（可分批暗示）。

**生成规则：**
1.  **主题驱动**：角色的欲望、恐惧、信念必须与核心命题相关联。
2.  **对手是主角的镜像**：对手的欲望与主角对立，但恐惧与主角同源。
3.  **配角功能明确**：每个配角必须有不可替代的故事功能（推动剧情、揭示主题、衬托主角）。
4.  **关系网密度优先**：先构建强关联的核心三角（主角-对手-关键盟友），再扩展外围。

**每个角色必须包含以下字段：**
- name: 姓名
- identity: 身份/职业
- role: 角色定位（protagonist/opponent/ally/mentor/supporting）
- desire: 核心欲望
- fear: 核心恐惧
- personality: 性格关键词（3-5个）
- arc: 人物弧光（开始时相信什么 → 结束时明白什么）

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "characters": [
        {
            "name": "姓名",
            "identity": "身份/职业",
            "role": "protagonist",
            "desire": "核心欲望",
            "fear": "核心恐惧",
            "personality": ["关键词1", "关键词2", "关键词3"],
            "arc": "人物弧光"
        }
    ],
    "relationship_map": {
        "nodes": [
            {"id": "唯一id", "name": "角色名", "role": "protagonist", "layer": "core"}
        ],
        "links": [
            {"source": "节点id", "target": "节点id", "type": "关系类型"}
        ]
    },
    "character_conflicts": ["核心角色之间的冲突点1", "冲突点2"]
}"""
        super().__init__(system_prompt)
    
    def format_output(self, content):
        """格式化输出角色卡片和关系图谱"""
        try:
            data = self._parse_json(content)
            output = "\n👥 **角色工坊生成结果**\n"
            
            # 角色卡片展示
            output += "\n## 角色卡片\n"
            for idx, char in enumerate(data.get("characters", []), 1):
                output += f"\n---\n**{idx}. {char.get('name', '')}**（{char.get('identity', '')}）\n"
                output += f"- **角色定位**：{char.get('role', '')}\n"
                output += f"- **核心欲望**：{char.get('desire', '')}\n"
                output += f"- **核心恐惧**：{char.get('fear', '')}\n"
                output += f"- **性格**：{'、'.join(char.get('personality', []))}\n"
                output += f"- **人物弧光**：{char.get('arc', '')}\n"
            
            # 关系图谱
            rel_map = data.get("relationship_map", {})
            if rel_map.get("links"):
                output += "\n## 关系图谱\n"
                for link in rel_map["links"]:
                    source = next((n["name"] for n in rel_map.get("nodes", []) if n["id"] == link["source"]), link["source"])
                    target = next((n["name"] for n in rel_map.get("nodes", []) if n["id"] == link["target"]), link["target"])
                    output += f"- {source} → {target}：{link.get('type', '')}\n"
            
            # 核心冲突
            conflicts = data.get("character_conflicts", [])
            if conflicts:
                output += "\n## 核心角色冲突\n"
                for c in conflicts:
                    output += f"- {c}\n"
            
            return output
            
        except Exception as e:
            print(f"格式化输出时出错: {e}")
            return content
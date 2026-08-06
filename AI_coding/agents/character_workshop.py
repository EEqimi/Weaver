# agents/character_workshop.py
import json
from agents.base_agent import BaseAgent

class CharacterWorkshop(BaseAgent):
    """人物工坊：基于主题与冲突，生成结构化的角色卡片和关系图谱"""
    
    def __init__(self):
        system_prompt = """你是一位资深的故事角色设计师，擅长根据故事主题和篇幅，创造有深度、有功能性的角色群像。

你的核心任务：
你将收到一个故事的完整上下文（包含设定、主题、核心冲突等）。请根据这些信息和指定的篇幅，生成一组结构完整的角色。

**生成规则：**
1.  **主题驱动**：角色的欲望、恐惧、信念必须与核心命题相关联。
2.  **对手是主角的镜像**：对手的欲望与主角对立，但恐惧与主角同源。
3.  **配角功能明确**：每个配角必须有不可替代的故事功能（推动剧情、揭示主题、衬托主角）。
4.  **关系网密度优先**：先构建强关联的核心三角（主角-对手-关键盟友），再扩展外围。
5.  **篇幅对应数量**：
    - 短篇：4人（1主角 + 2关键配角 + 1对立面）
    - 中篇：10人（2-3主角圈 + 4-6辅助圈 + 3-5背景圈）
    - 长篇：20人（3-5核心层 + 8-12次级层 + 10+流动层）

**输出格式：**
你必须严格按照以下JSON格式输出，不要包含任何其他文字。
{
    "characters": [
        {
            "name": "姓名",
            "identity": "身份/职业",
            "role": "protagonist | opponent | ally | mentor | supporting",
            "desire": "核心欲望",
            "fear": "核心恐惧",
            "personality": ["性格关键词1", "关键词2", "关键词3"],
            "arc": "人物弧光：开始时相信什么 → 结束时明白什么",
            "values": "价值观",
            "secret": "秘密",
            "key_experience": "关键经历",
            "relationships": "重要人际关系描述",
            "appearance": "标志性长相/穿着",
            "catchphrase": "口头禅",
            "mannerism": "小动作",
            "habit": "行为习惯"
        }
    ],
    "relationship_map": {
        "nodes": [
            {"id": "唯一id", "name": "角色名", "role": "protagonist", "layer": "core | secondary | background"}
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
                
                # 详细字段（折叠显示）
                details = []
                if char.get('values'): details.append(f"价值观：{char['values']}")
                if char.get('secret'): details.append(f"秘密：{char['secret']}")
                if char.get('key_experience'): details.append(f"关键经历：{char['key_experience']}")
                if char.get('appearance'): details.append(f"标志性外貌：{char['appearance']}")
                if details:
                    output += f"- **详情**：{'; '.join(details)}\n"
            
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
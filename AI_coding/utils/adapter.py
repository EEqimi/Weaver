# utils/adapter.py
import json
import re

from docx import Document
from openai import OpenAI

from utils.config import MODEL, get_client


def adapt_to_story_direction(user_description):
    """
    输入适配器：将用户的自由文本描述，解析为标准的StoryDirection格式
    """
    client = get_client()
    
    prompt = f"""你是一位资深的故事编辑，擅长从作者的描述中提炼出故事的核心要素。

用户提供了一段关于自己故事想法的描述（3-5段话）。请从这段描述中，提取或推断出以下七个维度，并以标准JSON格式输出。

用户描述：
{user_description}

输出格式：
{{
    "title": "根据描述推断的一个凝练标题",
    "unique_setting": "该故事世界最独特的一条规则或事实",
    "core_character": "中心人物的身份、核心欲望与核心恐惧",
    "main_conflict": "阻碍人物实现欲望的主要障碍",
    "possible_theme": "故事可能触及的深层主题（1-2个）",
    "symbolic_imagery": "一个可以贯穿故事的标志性意象",
    "hook_paragraph": "一段描绘故事开场或核心瞬间的引子"
}}

注意事项：
1. 如果描述中缺少某个维度，请根据上下文进行合理推断，不要留空
2. 所有推断必须基于用户描述，不能凭空创造无关内容
3. 输出需保持结构化，便于后续处理
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        # 清理可能的markdown标记
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())
        
    except Exception as e:
        print(f"适配器解析失败: {e}")
        raise ValueError("无法从你的描述中清晰理解故事要素，请补充：主角是谁？面临什么困境？")

def adapt_to_character_workshop(user_description):
    """
    人物工坊输入适配器：从用户自由文本描述中推断故事要素和篇幅
    """
    client = get_client()
    
    prompt = f"""你是一位资深的故事编辑。用户提供了一段关于故事想法的描述，请从中提取或推断出以下信息，并以JSON格式输出。

用户描述：
{user_description}

请推断：
1. story_direction：包含设定、核心人物、主要冲突、主题初探、标志性意象
2. theme_output：包含核心命题、情感基调、关键词体系
3. length：根据描述的复杂程度和涉及的事件数量，推断为'short'（短篇）、'medium'（中篇）或'long'（长篇）

输出格式：
{{
    "length": "short",
    "story_direction": {{
        "title": "推断的标题",
        "unique_setting": "独特设定",
        "core_character": "核心人物描述",
        "main_conflict": "主要冲突",
        "possible_theme": "可能主题",
        "symbolic_imagery": "标志性意象",
        "hook_paragraph": "引子段落"
    }},
    "theme_output": {{
        "core_theme": "核心命题",
        "emotional_tone": ["基调1", "基调2"],
        "keyword_system": {{
            "imagery_words": ["意象1", "意象2"],
            "emotion_words": ["情绪1", "情绪2"],
            "action_words": ["行动1", "行动2"]
        }}
    }}
}}

注意事项：
1. 所有推断必须基于用户描述，不能凭空创造无关内容
2. 如果某个维度信息不足，请根据上下文合理推断，不要留空
3. 输出需保持结构化，便于后续处理
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        # 清理可能的markdown标记
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())
        
    except Exception as e:
        print(f"人物工坊适配器解析失败: {e}")
        raise ValueError("我无法清晰理解你的故事要素，请补充：主角是谁？面临什么困境？")

def adapt_to_plot_architect(user_input):
    """
    情节建筑师输入适配器：从用户的故事前提中推断故事结构要素
    """
    client = get_client()

    prompt = f"""你是一位资深的故事编辑。用户提供了一个故事核心前提，请从中提取或推断出以下信息，并以JSON格式输出。

用户输入：
{user_input}

请推断：
1. 故事方向（StoryDirection）：包含标题、独特设定、核心人物（身份/欲望/恐惧）、主要冲突、可能主题、标志性意象。
2. 主题深化（ThemeOutput）：包含核心命题（一句话）、情感基调（3-4个词）、关键词体系（意象词/情绪词/行动词各3个）。
3. 篇幅（length）：根据描述的复杂程度，推断为'short'（短篇，建议5-10章）、'medium'（中篇，建议15-30章）或'long'（长篇，建议30-60章）。
4. 角色简表：至少包含主角和对手的姓名与核心欲望。

输出格式：
{{
    "story_direction": {{
        "title": "推断的标题",
        "unique_setting": "独特设定",
        "core_character": "核心人物描述",
        "main_conflict": "主要冲突",
        "possible_theme": "可能主题",
        "symbolic_imagery": "标志性意象",
        "hook_paragraph": "引子段落"
    }},
    "theme_output": {{
        "core_theme": "核心命题",
        "emotional_tone": ["基调1", "基调2", "基调3"],
        "keyword_system": {{
            "imagery_words": ["意象1", "意象2", "意象3"],
            "emotion_words": ["情绪1", "情绪2", "情绪3"],
            "action_words": ["行动1", "行动2", "行动3"]
        }}
    }},
    "length": "short/medium/long",
    "characters_brief": {{
        "protagonist": {{"name": "姓名", "desire": "欲望"}},
        "opponent": {{"name": "姓名", "desire": "欲望"}}
    }}
}}

注意事项：
1. 所有推断必须基于用户输入，不能凭空创造。
2. 如果某个维度信息不足，请根据上下文合理推断。
3. 输出需保持结构化，便于后续处理。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        # 清理可能的markdown标记
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())

    except Exception as e:
        print(f"情节建筑师适配器解析失败: {e}")
        raise ValueError("我无法清晰理解你的故事前提，请补充：主角是谁？核心冲突是什么？")

def parse_uploaded_file(file_content, file_type):
    """
    解析上传的文件，提取文本内容
    支持 .txt, .md, .docx
    """
    if file_type in ['txt', 'md']:
        return file_content.decode('utf-8')
    elif file_type == 'docx':
        doc = Document(file_content)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"不支持的文件格式: {file_type}")

def detect_continuation_mode(content):
    """
    检测用户上传的材料类型，判断是续写还是素材
    """
    # 简单启发式判断：检查是否有明显的章节标记（如"第X章"）
    chapter_pattern = r'第[一二三四五六七八九十百]+章'
    if re.search(chapter_pattern, content):
        return 'continuation'
    else:
        return 'material'

def extract_existing_chapters(content):
    """
    从已有文本中提取已写的章节
    """
    chapters = []
    # 按章节分割（简单实现，生产环境需更健壮）
    parts = re.split(r'(第[一二三四五六七八九十百]+章\s*[^\n]*)', content)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        chapters.append({
            "title": title,
            "content": body
        })
    return chapters
# knowledge_base/scripts/convert_md_to_json.py
import os
import re
import json

def parse_writer_block(text):
    """解析单个作家块，提取所有字段"""
    writer = {}
    
    # 提取姓名
    name_match = re.search(r'##\s*1\.\s*作家姓名\s*\n\s*(.+?)(?:\n|$)', text)
    if name_match:
        writer['name'] = name_match.group(1).strip()
    
    # 提取ID
    id_match = re.search(r'##\s*2\.\s*作家ID（拼音）\s*\n\s*(.+?)(?:\n|$)', text)
    if id_match:
        writer['writer_id'] = id_match.group(1).strip().lower().replace(' ', '_')
    
    # 提取时代
    era_match = re.search(r'##\s*3\.\s*时代\s*\n\s*(.+?)(?:\n|$)', text)
    if era_match:
        writer['era'] = era_match.group(1).strip()
    
    # 提取地区
    region_match = re.search(r'##\s*4\.\s*地区/国籍\s*\n\s*(.+?)(?:\n|$)', text)
    if region_match:
        writer['region'] = region_match.group(1).strip()
    
    # 提取体裁
    genres_match = re.search(r'##\s*5\.\s*主要体裁\s*\n\s*(.+?)(?:\n|$)', text)
    if genres_match:
        genres_str = genres_match.group(1).strip()
        writer['genres'] = [g.strip() for g in genres_str.split('、')]
    
    # 提取风格关键词
    tags_match = re.search(r'##\s*6\.\s*风格关键词（3-6个）\s*\n(.*?)(?:\n##\s*7\.|$)', text, re.DOTALL)
    if tags_match:
        tags_text = tags_match.group(1).strip()
        tags = re.findall(r'-\s*(.+?)(?:\n|$)', tags_text)
        writer['core_style_tags'] = [t.strip() for t in tags if t.strip()]
    
    # 提取详细风格描述
    desc_match = re.search(r'##\s*7\.\s*详细风格描述\s*\n(.*?)(?:\n##\s*8\.|$)', text, re.DOTALL)
    if desc_match:
        writer['style_description'] = desc_match.group(1).strip().replace('\n', ' ')
    
    # 提取代表作
    works_match = re.search(r'##\s*8\.\s*代表作（2-3部）\s*\n(.*?)(?:\n##\s*9\.|$)', text, re.DOTALL)
    if works_match:
        works_text = works_match.group(1).strip()
        works = re.findall(r'[-*]\s*《?(.+?)》?(?:\n|$)', works_text)
        writer['representative_works'] = [w.strip() for w in works if w.strip()]
    
    # 提取语料样本
    samples_match = re.search(r'##\s*9\.\s*语料样本（可选，1-2段原文）\s*\n(.*?)(?:\n##\s*10\.|$)', text, re.DOTALL)
    if samples_match:
        samples_text = samples_match.group(1).strip()
        samples = re.findall(r'>(.+?)(?:\n|$)', samples_text)
        writer['samples'] = [s.strip() for s in samples if s.strip()]
    
    # 提取一句话总结
    summary_match = re.search(r'##\s*10\.\s*风格描述用一句话总结（用于向量化检索）\s*\n\s*(.+?)(?:\n|$)', text)
    if summary_match:
        writer['embedding_description'] = summary_match.group(1).strip()
    
    return writer

def convert_md_to_json():
    # 从项目根目录定位 ref/writers.md
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    md_path = os.path.join(base_dir, 'ref', 'writers.md')
    output_dir = os.path.join(base_dir, 'knowledge_base', 'data', 'writers')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(md_path):
        print(f"❌ 文件不存在: {md_path}")
        return
    
    # 读取MD文件
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按"## 1. 作家姓名"分割
    blocks = re.split(r'(?=##\s*1\.\s*作家姓名)', content)
    
    count = 0
    for block in blocks:
        if not block.strip() or '作家姓名' not in block:
            continue
        
        writer = parse_writer_block(block)
        if writer and 'writer_id' in writer and 'name' in writer:
            filename = f"{writer['writer_id']}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(writer, f, ensure_ascii=False, indent=2)
            print(f"✅ 已生成: {filename} ({writer['name']})")
            count += 1
    
    print(f"\n🎉 转换完成！共生成 {count} 个JSON文件")
    print(f"📂 文件保存在: {output_dir}")

if __name__ == "__main__":
    convert_md_to_json()
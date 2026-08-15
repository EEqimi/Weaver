# knowledge_base/scripts/build_vector_store.py
import os
import sys
import json

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.knowledge_base import add_writer_to_db, collection, count_writers

def build_knowledge_base():
    """遍历 knowledge_base/data/writers/ 目录，将所有 JSON 文件导入向量库"""
    writer_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "writers")
    
    # 检查 writers 目录是否存在
    if not os.path.exists(writer_dir):
        print(f"❌ 目录不存在: {writer_dir}")
        print("请先创建 knowledge_base/data/writers/ 目录并放入作家 JSON 文件")
        return
    
    # 清空集合中的所有数据（保留集合本身）
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"🔄 已清空 {len(existing_ids)} 条旧数据")
        else:
            print("📭 知识库为空，准备导入")
    except Exception as e:
        print(f"⚠️ 清空数据时出错: {e}")
        print("🔄 尝试继续导入...")
    
    imported_count = 0
    json_files = [f for f in os.listdir(writer_dir) if f.endswith(".json")]
    
    if not json_files:
        print(f"❌ 在 {writer_dir} 中没有找到任何 .json 文件")
        return
    
    for filename in json_files:
        path = os.path.join(writer_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                writer_data = json.load(f)
                add_writer_to_db(writer_data)
                imported_count += 1
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析错误 {filename}: {e}")
        except Exception as e:
            print(f"❌ 导入失败 {filename}: {e}")
    
    print(f"\n🎉 知识库构建完成！共导入 {imported_count} 位作家")
    print(f"📊 当前知识库共有 {count_writers()} 条记录")

if __name__ == "__main__":
    build_knowledge_base()
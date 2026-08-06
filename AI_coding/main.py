# main.py
from agents.inspiration_catcher import InspirationCatcher


def main():
    catcher = InspirationCatcher()
    
    print("=" * 50)
    print("🧠 文思工坊 - 灵感捕捉器")
    print("输入你的想法，我会帮你生成故事方向")
    print("输入 '历史' 查看对话记录")
    print("输入 '清空' 开始新对话")
    print("输入 '退出' 结束程序")
    print("=" * 50)
    
    while True:
        user_input = input("\n💡 请输入你的想法: ")
        
        if user_input == "退出":
            print("👋 再见！")
            break
        elif user_input == "历史":
            print("\n📜 对话历史:")
            for msg in catcher.get_history():
                preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                print(f"  {msg['role']}: {preview}")
            continue
        elif user_input == "清空":
            catcher.clear_history()
            print("✅ 记忆已清空，开始新对话")
            continue
        
        print("\n🚀 正在处理...")
        result = catcher.run(user_input)
        formatted = catcher.format_output(result)
        print(formatted)

if __name__ == "__main__":
    main()
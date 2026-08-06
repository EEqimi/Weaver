# agents/base_agent.py
import json

from openai import OpenAI  # noqa: F401

from utils.config import MODEL, get_client


class BaseAgent:
    """所有智能体的基类，提供记忆和调用能力"""
    
    def __init__(self, system_prompt, model=MODEL):
        self.model = model
        self.client = get_client()
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
    
    def run(self, user_input):
        """接收用户输入，返回AI回复（带记忆）"""
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.6,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content
    
    def get_history(self):
        return self.messages
    
    def clear_history(self):
        self.messages = [self.messages[0]]
    
    def _parse_json(self, content):
        """解析JSON的通用方法"""
        clean = content.strip()
        if clean.startswith("```json"):  # noqa: FURB188
            clean = clean[7:]
        if clean.endswith("```"):  # noqa: FURB188
            clean = clean[:-3]
        return json.loads(clean.strip())
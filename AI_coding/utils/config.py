# utils/config.py
import streamlit as st
from openai import OpenAI

# 从 Streamlit Secrets 读取 API 配置
API_KEY = st.secrets["ALIYUN_API_KEY"]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# 创建全局客户端（所有节点共用）
def get_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

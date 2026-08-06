# utils/config.py
from openai import OpenAI

# API配置
API_KEY = "sk-ws-H.ERMYIEX.3N5K.MEQCIAWqpGL7P-sxRnft9j8urWd0yUOUyn2fQOvBb7rI7qHoAiADuasA9jgsUXdmey8wdprRAWh0KFjcp1RKofMBw_TwMA"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# 创建全局客户端（所有节点共用）
def get_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
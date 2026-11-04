import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

"""
# 加载.env
load_dotenv()

# 创建客户端
client = init_chat_model(   # 统一的方式
    model="deepseek-v4-flash",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)

# 访问llm
result = client.invoke('给我讲一个笑话')
# print(result)
print(result.content)
"""

"""
# 加载.env
load_dotenv()

# 创建客户端
client = ChatOpenAI(
    temperature=0.6,
    model="glm-5.3",
    openai_api_key=os.getenv('GLM_API_KEY'),
    openai_api_base=os.getenv('GLM_BASE_URL'),
)

# 访问llm
result = client.invoke('给我讲一个笑话')
# print(result)
print(result.content)

"""

# 加载.env
load_dotenv()

# 创建客户端
client = init_chat_model(   # 统一的方式
    model="glm-5.3",
    model_provider="openai",
    api_key=os.getenv('GLM_API_KEY'),
    base_url=os.getenv('GLM_BASE_URL'),
)

# 访问llm
result = client.invoke('给我讲一个笑话')
# print(result)
print(result.content)
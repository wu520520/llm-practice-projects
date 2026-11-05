# 加载.env
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

load_dotenv()

# 创建客户端
client = init_chat_model(   # 统一的方式
    model="deepseek-v4-flash",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)

# 访问llm-方式1  字符串
# result = client.invoke('给我讲一个笑话')

# 访问llm-方式2 字典
# result = client.invoke([
#     {"role": "system", "content": "你是一个笑话专家"},
#     {"role": "user", "content": "给我讲一个笑话"}
# ])

# 访问llm-方式3 BaseMessage
# result = client.invoke([
#     SystemMessage(content='你是一个笑话专家'),
#     HumanMessage(content='给我讲一个笑话')
# ])
# 访问llm-方式4 元组
result = client.invoke([
    ("system", "你是一个笑话专家"),
    ("user", "给我讲一个笑话")
])

print(3333333333333333)
print(result.content)
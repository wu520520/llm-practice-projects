# ainvoke 异步调用
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


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
result = client.stream('给我讲一个笑话')

for chunk in result:
    print(chunk.content, end='')





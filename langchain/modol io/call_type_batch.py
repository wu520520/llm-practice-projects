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
# result = client.batch(inputs=[
#     "给我讲一个笑话",
#     "什么是LangChain？"
# ])

result = client.batch(inputs=[
    [('system', '你是一个笑话专家'), ('user', '给我讲一个笑话')],
    [('system', '你是一个IT专家'), ('user', '什么是LangChain？')],
])

for res in result:
    print(res.content)







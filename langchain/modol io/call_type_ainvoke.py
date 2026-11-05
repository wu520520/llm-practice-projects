# ainvoke 异步调用
import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

async def call_llm():
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
    result = await client.ainvoke('给我讲一个笑话')

    print(result)


def print_3():
    print(33333333333333333333)



if __name__ == '__main__':
    asyncio.run(call_llm())
    print_3()

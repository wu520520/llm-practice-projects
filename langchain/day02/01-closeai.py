import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 创建客户端
client = OpenAI(
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url=os.getenv("CLOSEAI_BASE_URL"),
    api_key=os.getenv("CLOSEAI_API_KEY"),
)

# 调用模型
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say hi",
        }
    ],
    model="gpt-4o-mini",  # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
)

print(chat_completion)
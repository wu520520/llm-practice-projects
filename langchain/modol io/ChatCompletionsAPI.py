import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载.env文件
load_dotenv()


# 创建客户端
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)

# 使用客户端访问大模型
result = client.chat.completions.create(
    model='deepseek-v4-pro',
    messages=[{"role": "system", "content": "你是一位资深的笑话专家"},
              {"role": "user", "content": "帮忙生成一个关于程序员的笑话"}]
)

# print(result)
print(result.choices[0].message.content)
print(result.choices[0].message.reasoning_content)
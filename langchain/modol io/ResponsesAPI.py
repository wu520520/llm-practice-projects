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
result = client.responses.create(model='deepseek-v4-flash', input='你是谁')

print(result)

print("推理--》", result.output[0].content[0].text)

print("输出--》", result.output[1].content[0].text)

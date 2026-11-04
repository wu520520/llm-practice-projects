"""
需求: 让大模型找出中国最重要的节日及节日的描述

数据必须的json格式，格式如下:{festival:['春节','中秋节'], desc:['春节是中国最重要的节日,.....]}

"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# 定义json schema
class Festival(BaseModel):
    festivals: list[str] = Field(description="中国最重要的节日")
    descriptions: list[str] = Field(description="节日的描述")

# 加载环境变量
load_dotenv()

# 创建客户端
client = OpenAI(
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url=os.getenv("CLOSEAI_BASE_URL"),
    api_key=os.getenv("CLOSEAI_API_KEY"),
)

# 调用模型
chat_completion = client.chat.completions.parse(
    messages=[
        {
            "role": "user",
            "content": "找出中国最重要的节日及节日的描述",
        }
    ],
    model="gpt-4o-mini",  # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动,
    response_format=Festival,
)

print(chat_completion)
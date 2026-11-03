import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 创建大模型客户端
llm = OpenAI(
    api_key=os.getenv("CLOSEAI_API_KEY"),
    base_url=os.getenv("CLOSEAI_BASE_URL")
)

# 声明工具函数
def get_weather(city, date):
    return f"{city} on {date} is cloudy with a chance of rain."

# 工具描述
tools = [
    {
        "type": "function",
        "function": {
              "name": "get_weather",
              "description": "Get today's weather for a location.",
              "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称, e.g. San Francisco",},
                        "date": {"type": "string", "description":"想要查询的天气的日期, e.g. 2023-12-25"}
                    },
                    "required": ["city", "date"],
                    "additionalProperties": False,
              },
              "strict": True,
        }
    }
]

# 提示词
messages = [
    {"role": "user", "content": "What is the weather like in 北京 on 2026-12-12 ?"}
]

# 第一次调用模型
first_response = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

# print(first_response.choices[0].message.tool_calls[0])
# print(first_response.choices[0].message.tool_calls[0].function)

# 将大模型的第一回复的消息添加到提示词
messages.append(first_response.choices[0].message)

# 手动调用工具
for tool_call in first_response.choices[0].message.tool_calls or []:
    if tool_call.function.name == 'get_weather':
        args = json.loads(tool_call.function.arguments)
        weather = get_weather(args["city"], args["date"])
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"weather": weather}),
        })


# 第二次调用大模型
second_response = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

print(second_response)
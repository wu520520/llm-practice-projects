import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 创建大模型客户端
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("CLOSEAI_API_KEY"),
    base_url=os.getenv("CLOSEAI_BASE_URL")
)

from pydantic import BaseModel,Field
class GetWeatherArgs(BaseModel):
    city: str = Field(description="城市名称")
    date: str = Field(description="日期，格式为YYYY-MM-DD")
# 声明工具
@tool(description="获取指定城市在指定日期的天气", args_schema=GetWeatherArgs)
def get_weather(city, date):
    return f"{city} on {date} is cloudy with a chance of rain."


# 将工具绑定到llm上并返回新的llm
llm = llm.bind_tools([get_weather])

# 提示词
messages = [
    HumanMessage(content="北京2026-12-25的天气")
]

# 第一次调用模型
first_response = llm.invoke(messages)


# 将大模型的第一回复的消息添加到提示词
messages.append(first_response)

# 手动调用工具
# 解析调用工具的出参，手动调用工具
tool_call = first_response.tool_calls[0]
args = tool_call['args']
call_tool_res = get_weather.invoke(args)
id = tool_call['id']

#构造Message对象
tool_message = ToolMessage(tool_call_id=id, content=call_tool_res)
messages.append(tool_message)

# 第二次调用大模型
second_response = llm.invoke(messages,)

print(second_response)
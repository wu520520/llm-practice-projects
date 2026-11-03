import os

from dotenv import load_dotenv
from langchain.agents import create_agent
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

# 创建Agent
agent = create_agent(
    model=llm,
    tools=[get_weather]
)

result = agent.invoke(input={
    'messages': [{'role': 'user', 'content':'北京2026-12-25的天气'}]
})
print(result)
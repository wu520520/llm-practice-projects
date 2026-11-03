import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# 加载环境变量
load_dotenv()

# 创建大模型客户端
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("CLOSEAI_API_KEY"),
    base_url=os.getenv("CLOSEAI_BASE_URL")
)


# 定义 Tavily 搜索工具
search = TavilySearch(max_results=5)

# 创建Agent
agent = create_agent(
    model=llm,
    tools=[search]
)

# result = agent.invoke(input={
#     'messages': [{'role': 'user', 'content': '北京2026-9-11的天气'}]
# })


# 调用 Agent
for chunk in agent.stream({"messages": [
                          {"role": "system", "content": "你是位助手，需要调用工具来帮助用户。"},
                          {"role": "user", "content": "今天北京的天气怎么样？"},]}):

    if chunk.get('model') is not None:
        print(chunk['model']['messages'][0].content, end="\n\n")
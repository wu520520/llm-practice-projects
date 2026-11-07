import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI


async def call_12306():
    # 创建
    mcp_client = MultiServerMCPClient(
        {
            "12306-mcp": {
                "transport": "streamable_http",
                "url": "https://mcp.api-inference.modelscope.net/ff923025cdbc4e/mcp"
            }
        }
    )

    # 获取McpServer中所提供的工具
    tools = await mcp_client.get_tools()
    # print(tools)

    # 创建大模型客户端
    # 加载环境变量
    load_dotenv()
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=os.getenv("CLOSEAI_API_KEY"),
        base_url=os.getenv("CLOSEAI_BASE_URL")
    )

    # 创建智能体
    agent = create_agent(model=llm, tools=tools)

    # 运行智能体
    result = await agent.ainvoke(input={
        "messages": [
            {"role": "user", "content": "查询今天从北京到上海的动车票,以表格的方式输出结果"}
        ]
    })

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(call_12306())
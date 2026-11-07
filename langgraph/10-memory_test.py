import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("CLOSEAI_API_KEY"),
    base_url=os.getenv("CLOSEAI_BASE_URL")
)

# 创建短期记忆
checkpointer = InMemorySaver()
summary_middleware = SummarizationMiddleware(model=llm,trigger=("messages", 10))
agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer,
    middleware=[summary_middleware]
)

res1 = agent.invoke(
    input={
        "messages": [
            {"role": "user", "content": "你好,我叫张三"}
        ]
    },
    config={"configurable": {"thread_id": "123"}}
)
print(f'第一次回答:{res1}')

res2 = agent.invoke(
    input={
        "messages": [
            {"role": "user", "content": "我叫什么名字?"}
        ]
    },
    config={"configurable": {"thread_id": "123"}}
)
print(f'第二次回答:{res2}')

res3 = agent.invoke(
    input={
        "messages": [
            {"role": "user", "content": "什么是LangChain？"}
        ]
    },
    config={"configurable": {"thread_id": "123"}}
)
print(f'第三次回答:{res3}')


res4 = agent.invoke(
    input={
        "messages": [
            {"role": "user", "content": "我刚才问了什么问题?"}
        ]
    },
    config={"configurable": {"thread_id": "123"}}
)
print(f'第四次回答:{res4}')
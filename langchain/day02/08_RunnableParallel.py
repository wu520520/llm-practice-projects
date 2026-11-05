import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# 加载环境
load_dotenv()

# 创建客户端
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 英文翻译链
en_chain = PromptTemplate.from_template("把这个句子{topic}翻译成英文") | llm | StrOutputParser()

# 发文翻译链
fr_chain = PromptTemplate.from_template("把这个句子{topic}翻译成法文") | llm | StrOutputParser()

# 将两个链组合成并行链
par_chain = RunnableParallel(english=en_chain, franch=fr_chain)

# 调用并行链
result = par_chain.invoke({"topic": "什么是人工智能?"})

print(result)

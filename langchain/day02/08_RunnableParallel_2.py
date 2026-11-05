import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# 加载环境
load_dotenv()

# 创建客户端
deepseek_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

glm_llm = ChatOpenAI(
    model="glm-5.3-flash",
    base_url=os.getenv("GLM_BASE_URL"),
    api_key=os.getenv("GLM_API_KEY"),
)

gpt_llm = ChatOpenAI(
    model="gpt-5.6-luna",
    base_url=os.getenv("CLOSEAI_BASE_URL"),
    api_key=os.getenv("CLOSEAI_API_KEY"),
)




# deepseek评价链
deepseek_chain = PromptTemplate.from_template("对这首诗{poem}做一下赏析，分析它蕴含的含义") | deepseek_llm | StrOutputParser()

# glm评价链
glm_chain = PromptTemplate.from_template("对这首诗{poem}做一下赏析，分析它蕴含的含义") | glm_llm | StrOutputParser()

# 对比链
gpt_chain = PromptTemplate.from_template("这两种赏析，第一种：{paragraph_1}，第二种：{paragraph_2}，哪个更好，为什么") | gpt_llm | StrOutputParser()

# 组合链
map_chain = {
    "paragraph_1": deepseek_chain,
    "paragraph_2": glm_chain,
} | gpt_chain

# 调用组合连
result = map_chain.invoke({"poem": """
     菩提本无树，
     明镜亦非台，
     本来无一物，
     何处惹尘埃。
     """})

print(result)



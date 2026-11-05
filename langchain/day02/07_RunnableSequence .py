
"""
给某个电影演员进行演技上的评分
"""
from langchain_core.runnables import RunnableSequence

"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1.加载环境
load_dotenv()

# 2.创建客户端
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 3.创建提示词模板
prompt_template = PromptTemplate.from_template('你是一位专业的演员评论家,请对{actor}的演技做出评价,限制500字以内，如果不知道该演员返回我不知道')

# 4.格式化提示词模板
prompt = prompt_template.invoke({
    "actor": "刘德华"
})

# 5.调用模型
result = llm.invoke(prompt)

# 6.创建字符串解析器
str_parser = StrOutputParser()

# 7.调用字符串解析器将模型返回的结果转换为字符串
final_result = str_parser.invoke(result)

print(final_result)

"""


import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1.加载环境
load_dotenv()

# 2.创建客户端
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
# 3.创建提示词模板
prompt_template = PromptTemplate.from_template('你是一位专业的演员评论家,请对{actor}的演技做出评价,限制500字以内，如果不知道该演员返回我不知道')

# 4.创建字符串解析器
str_parser = StrOutputParser()

# 5.将各个组件组成chain
# chain = prompt_template | llm | str_parser
chain= RunnableSequence(prompt_template, llm, str_parser)
# 6.执行chain
result = chain.invoke({'actor': '刘德华'})
print(result)
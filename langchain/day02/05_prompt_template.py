"""
  langchain提供了两种提示词模板
  1.PromptTemplate --> 字符串模板   你是专业的影评人,请对{movie}做出影评
  2.ChatPromptTemplate --> 聊天模板
   --> 可以是字符串模板  你是专业的影评人,请对{movie}做出影评
   --> 消息类型模板
       invoke([
        SystemMessage(content="你是专业的影评人"),
        HumanMessage(content="请对{movie}做出影评"),
       ])
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 创建客户端
client = ChatOpenAI(
    model="gpt-4o-mini",
    base_url = os.getenv("CLOSEAI_BASE_URL"),
    api_key = os.getenv("CLOSEAI_API_KEY"),
)

# 创建提示词模板
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("你是专业的影评人,请对{movie}做出影评, 字数限制在500以内,如果该电影你没有听说过直接回答我不知道")

while True:
    if input('是否继续？(y/n):') != 'y':
        break
    movie_name = input('请输入要进行影评的电影名称:')
    # 格式化提示词模板
    prompt = prompt_template.invoke({"movie": movie_name})
    # 调用模型
    result = client.invoke(prompt)
    print(result.content)



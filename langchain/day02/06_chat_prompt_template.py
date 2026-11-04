"""
  langchain提供了两种提示词模板
  1.PromptTemplate --> 字符串模板   你是专业的影评人,请对{movie}做出影评
  2.ChatPromptTemplate --> 聊天模板
   --> 可以是字符串模板  你是专业的影评人,请对{movie}做出影评
   --> 消息类型模板
       invoke([
        ('system',"你是专业的影评人"),
        ('haman',请对{movie}做出影评"),
       ])


   SystemMessage、HumanMessage  不能有变量的
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 创建客户端
client = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    api_key = os.getenv("DEEPSEEK_API_KEY"),
)

# 创建提示词模板
from langchain_core.prompts import ChatPromptTemplate
# prompt_template = ChatPromptTemplate.from_template("你是专业的影评人,请对{movie}做出影评, 字数限制在500以内,如果该电影你没有听说过直接回答我不知道")
prompt_template = ChatPromptTemplate.from_messages(messages=[
    # ('system', "你是一个专业的商品评论员,擅长对电子产品进行评测,请对用户输入的产品给出专业的评价,字数限制在500以内,如果产品不存在返回我不知道"),
    SystemMessage(content="你是一个专业的商品评论员,擅长对电子产品进行评测,请对用户输入的产品给出专业的评价,字数限制在500以内,如果产品不存在返回我不知道"),
    ('human', "请评价{product}的优缺点，包括{aspect1}和{aspect2}。"),
])

while True:
    if input('是否继续？(y/n):') != 'y':
        break
    product = input('请输入要进行评价的商品名称:')
    aspect1 = input('请输入评价的角度1:')
    aspect2 = input('请输入评价的角度2:')
    # 格式化提示词模板
    prompt = prompt_template.invoke({
        "product": product,
        "aspect1": aspect1,
        "aspect2": aspect2,
    })
    print(prompt)
    # 调用模型
    result = client.invoke(prompt)
    print(result.content)



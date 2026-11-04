
from langchain_ollama import ChatOllama

# 创建客户端
client = ChatOllama(model='deepseek-r1:7b')

# 调用模型
result = client.invoke('张家口今天天气怎么样?')
print(result.content)

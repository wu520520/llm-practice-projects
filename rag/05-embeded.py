from langchain_huggingface import HuggingFaceEmbeddings

# 加载嵌入模型
embeded_model = HuggingFaceEmbeddings(model_name='./model/bge-base-zh-v1.5')


# 进行语义嵌入
# result = embeded_model.embed_query("什么是人工智能")

result = embeded_model.embed_documents(['什么是人工智能', '什么是LangChain'])
print(result)



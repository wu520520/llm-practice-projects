from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 创建md文件加载器
# md_loader = UnstructuredMarkdownLoader(file_path='./data/sample.md', mode='single')
md_loader = UnstructuredMarkdownLoader(file_path='./data/sample.md', mode='elements')

# 加载文件
result = md_loader.load()

for doc in result:
    print(doc)
    print('*' * 100)

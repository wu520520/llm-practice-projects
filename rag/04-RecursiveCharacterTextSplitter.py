from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
doc_list = UnstructuredWordDocumentLoader(file_path='./data/sample.docx', mode='single').load()

# 定义切分器
spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。"]
)

# 对文档进行切分
chunks = spliter.split_documents(doc_list)

for chunk in chunks:
    print(chunk)
    print('*' * 50, len(chunk.page_content), '*' * 50)
import logging

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO)
def get_milvus_client():
    """创建mivus的客户端"""
    return MilvusClient(uri='http://localhost:19530')


def insert_data(client:MilvusClient, collection_name:str):
    """插入数据
    1.加载数据
    2.数据切片
    3.获取稠密和稀疏的向量
    4.写入milvus
    """
    # 1.加载数据
    doc = UnstructuredWordDocumentLoader(file_path='./data/sample.docx.', mode='single').load()
    # 2.数据切片
    # 定义切分器
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。"]
    )

    # 对文档进行切分
    chunks = spliter.split_documents(doc)[0:20]

    # 3.获取稠密和稀疏的向量
    from FlagEmbedding import BGEM3FlagModel
    embeded_model = BGEM3FlagModel(model_name_or_path='./model/bge-m3')
    sentences = [chunk.page_content for chunk in chunks]
    all_vectors = embeded_model.encode(sentences=sentences, return_dense=True, return_sparse=True)

    # 写入milvus
    data_list = []
    for chunk, dense_vec, sparse_vec in zip(chunks, all_vectors['dense_vecs'], all_vectors['lexical_weights']):
        data_list.append({
           "text": chunk.page_content,
           "vector": dense_vec,
           "sparse_vector": sparse_vec,
           "metadata": chunk.metadata
        })
    cnt = client.insert(collection_name=collection_name, data=data_list)
    print(f'成功插入{cnt}条数据')



if __name__ == '__main__':
    client = get_milvus_client()
    insert_data(client, 'demo_collection')
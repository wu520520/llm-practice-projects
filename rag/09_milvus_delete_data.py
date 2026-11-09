import logging

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO)
def get_milvus_client():
    """创建mivus的客户端"""
    return MilvusClient(uri='http://localhost:19530')


def delete_data(client:MilvusClient):
    res = client.delete(
                collection_name = "demo_collection",
                 # 过滤条件，仅删除 id 在指定范围内的实体,也可以传递其他字段的过滤条件
                filter="id in [468981664801065864, 468981664801065865]",
          )
    print(res)



if __name__ == '__main__':
    client = get_milvus_client()
    delete_data(client)
"""
Milvus Vector Search:
    1.查询稠密向量
    2.查询稀疏向量
    3.混合查询

Milvus Constant Search:
    1.查询常量

"""
from pymilvus import MilvusClient, RRFRanker, AnnSearchRequest


def get_milvus_client():
    """创建mivus的客户端"""
    return MilvusClient(uri='http://localhost:19530')


def encode_query(query: str):
    from FlagEmbedding import BGEM3FlagModel
    embeded_model = BGEM3FlagModel(model_name_or_path='./model/bge-m3')
    all_vectors = embeded_model.encode(sentences=query, return_dense=True, return_sparse=True)
    return all_vectors['dense_vecs'], all_vectors['lexical_weights']



def search_milvus_dense(client: MilvusClient, dense_vec):
    """向量查询"""
    result = client.search(
        collection_name="demo_collection",  # 查询哪个collection
        data=[dense_vec],
        anns_field="vector",
        limit=3,
        output_fields=['id', 'text', 'metadata'],
        search_params={"metric_type": "L2"}
    )

    print(result)

def search_milvus_sparse(client: MilvusClient, sparse_vec):
    """关键词查询"""
    result = client.search(
        collection_name="demo_collection",  # 查询哪个collection
        data=[sparse_vec],
        anns_field="sparse_vector",
        limit=3,
        output_fields=['id', 'text', 'metadata'],
        search_params={"metric_type": "IP"}
    )

    print(result)


def search_milvus_hybrid(client: MilvusClient, dense_vec, sparse_vec):
    """混合查询"""
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="vector",
        limit=3,
        param={"metric_type": "L2"}
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        limit=3,
        param={"metric_type": "IP"}
    )

    result = client.hybrid_search(
       collection_name='demo_collection',
       reqs=[dense_req, sparse_req],
       ranker=RRFRanker(),
       limit=3,
       output_fields=['id', 'text', 'metadata'],
    )

    return result



def text_query(client: MilvusClient, query):
    result = client.query(
        collection_name='demo_collection',
        filter=f"text like '%{query}%'",
        output_fields=['id', 'text', 'metadata'],
    )
    print(result)

if __name__ == '__main__':
    client = get_milvus_client()
    dense_vector, sparse_vector = encode_query('监护人应该履行什么责任?')
    # search_milvus_dense(client, dense_vector)
    # search_milvus_sparse(client, sparse_vector)
    # text_query(client, '监护人履行监护职责')
    result = search_milvus_hybrid(client, dense_vector,sparse_vector)

    # for entity in result[0]:
    #     print(entity['entity']['text'])
    #     print('*' * 50)

    context = "\n".join([entity['entity']['text'] for entity in result[0]])
    print(context)

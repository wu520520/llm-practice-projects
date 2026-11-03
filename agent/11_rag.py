import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


def get_milvus_client():
    """创建mivus的客户端"""
    return MilvusClient(uri='http://localhost:19530')

def encode_query(query: str):
    from FlagEmbedding import BGEM3FlagModel
    embeded_model = BGEM3FlagModel(model_name_or_path='./model/bge-m3')
    all_vectors = embeded_model.encode(sentences=query, return_dense=True, return_sparse=True)
    return all_vectors['dense_vecs'], all_vectors['lexical_weights']

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


def rag(query):
    # 加载环境
    load_dotenv()

    # 模型客户端
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=os.getenv("CLOSEAI_API_KEY"),
        base_url=os.getenv("CLOSEAI_BASE_URL"),
    )

    # 获取milvus客户端
    milvus_client = get_milvus_client()

    # 对query进行编码
    dense_vecs, sparse_vecs = encode_query(query)

    # 查询上下文
    result = search_milvus_hybrid(milvus_client, dense_vecs, sparse_vecs)

    # 构建上下文
    context = "\n".join([entity['entity']['text'] for entity in result[0]])

    # 构建提示词
    message_list = [
          {"role": "system","content": "你是一个专业的法律问答机器人，请根据上下文回答问题，当上下文无法回答问题时，请回答“根据上下文无法回答该问题”"},
          {"role": "user", "content": f"根据以下上下文回答问题：{context}\n问题：{query}"}
         ]
    response = llm.invoke(message_list)
    print(response.content)


if __name__ == '__main__':
    rag('监护人应该履行什么责任')
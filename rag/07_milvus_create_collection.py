from pymilvus import MilvusClient, DataType


def get_milvus_client():
    """创建mivus的客户端"""
    return MilvusClient(uri='http://localhost:19530')

def create_schma():
    """创建milvus collection结构"""
    return (MilvusClient.create_schema(auto_id=True)
        .add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        ).add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=1500,
        ).add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024,
        ).add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        ).add_field(
            field_name="metadata",
            datatype=DataType.JSON,
        ))


def create_index():
    """创建milvus index"""
    params = MilvusClient.prepare_index_params()
    # 给稠密向量添加分层导航小世界索引
    params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
    )
    params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    return params


"""
  id 自增   text 原文文本  vector 稠密向量  sparse_vector 稀疏向量  metadata 元数据
"""
def create_collection(client:MilvusClient):
    client.create_collection(
        collection_name="demo_collection",
        schema=create_schma(),
        index_params=create_index()
    )

    print(client.list_collections())
    print(client.describe_collection(collection_name='demo_collection'))




if __name__ == '__main__':
    client = get_milvus_client()
    create_collection(client)
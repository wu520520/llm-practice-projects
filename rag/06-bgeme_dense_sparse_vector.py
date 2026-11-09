from FlagEmbedding import BGEM3FlagModel
# 记载模型
embeded_model = BGEM3FlagModel(model_name_or_path='./model/bge-m3')


# 语义嵌入和关键词嵌入 {'dense_vecs':[[-0.01832025, -0.00965298,....],[0.00241487, -0.01131663,...]], 'lexical_weights':[[0.1, 0.2, ...], [0.3, 0.4, ...]]}
res = embeded_model.encode(sentences=['什么是人工智能', '什么是AI'], return_dense=True, return_sparse=True)

print('稠密向量:', res['dense_vecs'])
print('*' * 100)
print('稀疏向量:', res['lexical_weights'])
print('*' * 100)
print('稀疏向量转化为token:', embeded_model.convert_id_to_token(res['lexical_weights']))
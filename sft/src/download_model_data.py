
# 下载模型 Qwen3-0.6B-Base
# 模型下载
"""
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B-Base', cache_dir='.')
print(f'模型位置:{model_dir}')
"""

# 数据下载 modelscope download --dataset HuggingFaceH4/ultrachat_200k  --local_dir .
"""
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('HuggingFaceH4/ultrachat_200k', data_dir='.')
print(f'数据位置:{ds}')
"""
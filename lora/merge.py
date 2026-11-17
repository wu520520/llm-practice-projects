from argparse import ArgumentParser

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1 定义命令行参数
parser = ArgumentParser()
parser.add_argument('--base_model', type=str, help='原始模型的路径')
parser.add_argument('--peft_model', type=str, help='适配器模型的路径')
parser.add_argument('--merged_model', type=str, help='合并后模型的路径')
args = parser.parse_args()

# 根据参数获取原始模型、lora旁支、合并后的路径
base_model_path = args.base_model
peft_model_path = args.peft_model
merged_model_path = args.merged_model

# 2 加载原始模型和lora旁支分词器
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

# 3 将原始模型添加lora旁支结构形成新的模型结构
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

# 4 将原始模型参数和lora参数合并
merged_model = peft_model.merge_and_unload()

# 5 保存
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)


"""
python merge.py --base_model ./model/Qwen3-0.6B --peft_model ./model/Qwen3-0.6B-sft-lora --merged_model ./model/Qwen3-0.6B-merged
"""


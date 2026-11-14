from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 定义命令行参数
parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen3-0.6B', help='该参数是模型位置', required=True)
parser.add_argument('--prompt', type=str, default='123', help='该参数是提示词', required=True)

# 2.解析参数
args = parser.parse_args()

# 3.加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# 4.处理用户数据
prompt = args.prompt
message_list = [{"role": "user", "content": prompt}]
result = tokenizer.apply_chat_template(message_list, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
input_ids = result["input_ids"].to('cuda')
attention_mask = result["attention_mask"].to('cuda')
# 5.让模型自回归生成
generated_text = model.generate(inputs=input_ids, attention_mask=attention_mask, max_new_tokens=500, eos_token_id=[151643, 151645])
token_ids = generated_text[0]
response = tokenizer.decode(token_ids[len(input_ids[0]):], skip_special_tokens=True)
print(response)


"""
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B  --local_dir ./model/Qwen3-0.6B

python predict.py  --model_name ./model/Qwen3-0.6B-merged  --prompt "对新疆冰雪旅游的现状及开发可行性进行分析,归纳出影响新疆冰雪旅游的影响因素:淡旺季游客不均衡;交通不便;市场不成熟;缺乏专业人才;宣传力度不够.提出新疆冰雪旅游发展对策:发挥政府主管部门的宏观调控作用;加大资金投入;树立品牌,提高竞争力;利用发达的媒体进行宣传;培养相关人才.
识别出上文中的关键词"
"""




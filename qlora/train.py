from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import os

from trl import SFTConfig, SFTTrainer
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs/qlora"


# 1 处理数据
# 1.1 读取并划分数据
import torch
from datasets import load_dataset
data = load_dataset('json', data_files={'train': './data/psychology_data.jsonl'})
data = data['train'].train_test_split(test_size=0.2)

# 1.2 数据的批处理函数
def convert_function(example):
    conversation_list = example['conversation']
    massage_list = []
    for data in conversation_list:
        conversation_dict = data[0]
        current_data_massage_list = [
            {'role': 'user', 'content': conversation_dict['human']},
            {'role': 'assistant', 'content': conversation_dict['assistant']}
        ]
        massage_list.append(current_data_massage_list)

    return {'messages': massage_list}


# 1.3 调用map方法对数据进行批处理
dataset_dict = data.map(convert_function, batched=True, remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])


# 2 对模型进行量化
# 2.1 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
)

# 2.2 加载模型并进行量化
model = AutoModelForCausalLM.from_pretrained('model/Qwen3-8B', quantization_config=quantization_config)
# 2.3 准备模型进行kbit训练    1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm head to fp32 4- Freezing the base model layers to ensure they are not updated during training
quantized_model = prepare_model_for_kbit_training(model)

# 3 lora微调
# 3.1 定义lora配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate", "up_proj", "down_proj"],
    # target_modules=["q_proj", "v_proj" ],
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# 3.2 根据lora配置给模型添加旁支
peft_model = get_peft_model(quantized_model, lora_config)


# 4 训练模型
# 4.1 配置类
sft_config = SFTConfig(
    # 数据规模相关
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=500,
    # 训练日志相关
    logging_strategy='steps',
    logging_steps=10,
    report_to='tensorboard',
    # 学习率和优化器
    learning_rate=3.0e-4,  # 注意:对于lora微调来说,因为跟新的参数量较少,所以学习率可以设置大一些
    lr_scheduler_type='cosine',
    warmup_steps=0.1,
    optim='paged_adamw_32bit',
    # 评估和保存相关
    eval_strategy='steps',
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=3,
    output_dir='/root/autodl-tmp/qlora/checkpoints',  # 保存检查点，日志等信息
    # 优化相关
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=650,
    # 其他参数
    assistant_only_loss=True,
    chat_template_path='test_chat_template.jinja'
)

# 2.2 构造SFTTrainer
tokenizer = AutoTokenizer.from_pretrained('./model/Qwen3-8B')
sft_trainer = SFTTrainer(model=peft_model,
                         processing_class=tokenizer,
                         train_dataset=dataset_dict['train'],
                         eval_dataset=dataset_dict['test'],
                         args=sft_config)


# 2.3 开始训练
sft_trainer.train()

# 2.4 保存模型
sft_trainer.save_model('model/Qwen3-8B-qlora')


"""
modelscope download --model Qwen/Qwen3-8B  --local_dir ./model/Qwen3-8B
"""
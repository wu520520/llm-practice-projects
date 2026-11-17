from datasets import load_dataset
from unsloth import FastLanguageModel  # 必须放在第一行
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs/unsloth_qlora"

# 1 加载模型和分词器(量化)
quntiled_model, tokenizer = FastLanguageModel.from_pretrained(
    # 想让Unsloth加载本地模型，需要使用：./xxxx 这种方式来写路径，不能直接写 model/Qwen3-8B
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,  # 使用4位精度加载模型(量化)
    # 额外添加下面两个参数，让Unsloth只加载本地模型的参数其他配置文件
    use_exact_model_name=True,
    local_files_only=True
)

# 2.添加lora旁支
peft_model = FastLanguageModel.get_peft_model(
    model=quntiled_model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)


# 3. 处理数据
data = load_dataset('json', data_files={'train': './data/psychology_data.jsonl'})
data = data['train'].train_test_split(test_size=0.2)
def convert_function(example):
    conversation_list = example['conversation']
    text_list = []
    for data in conversation_list:
        conversation_dict = data[0]
        current_data_massage_list = [
            {'role': 'user', 'content': conversation_dict['human']},
            {'role': 'assistant', 'content': conversation_dict['assistant']}
        ]
        text_list.append(tokenizer.apply_chat_template(current_data_massage_list, tokenize=False))

    return {'text': text_list}

dataset_dict = data.map(convert_function, batched=True, remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])


# 4 训练模型
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
    output_dir='/root/autodl-tmp/unsloth/checkpoints',  # 保存检查点，日志等信息
    # 优化相关
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=650,
    # 其他参数
    # assistant_only_loss=True,
    # chat_template_path='test_chat_template.jinja'
)

# 2.2 构造SFTTrainer
sft_trainer = SFTTrainer(model=peft_model,
                         processing_class=tokenizer,
                         train_dataset=dataset_dict['train'],
                         eval_dataset=dataset_dict['test'],
                         args=sft_config)

# 计算模型生成部分的损失
from unsloth.chat_templates import train_on_responses_only
sft_trainer = train_on_responses_only(
    trainer=sft_trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)

# 2.3 开始训练
sft_trainer.train()

# 2.4 保存模型(只保存旁支)
# sft_trainer.save_model('model/Qwen3-8B-qlora')

# 保存合并模型
peft_model.save_pretrained_merged("./model/Qwen3-8B-merged", tokenizer, save_method="merged_16bit")
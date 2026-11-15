from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


# 0 准备工作
model = AutoModelForCausalLM.from_pretrained('model/Qwen3-0.6B')
tokenizer = AutoTokenizer.from_pretrained('model/Qwen3-0.6B')


# 1 数据处理
# 1.1 加载数据
dataset_dict = load_dataset('json', data_files={'train': 'data/keywords_data_train.jsonl',
                                                'test': 'data/keywords_data_test.jsonl'})
print(dataset_dict)


# 1.2 数据转换函数
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
dataset_dict = dataset_dict.map(convert_function, batched=True,
                                remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])


# 2 模型训练

# 2.1 配置类
sft_config = SFTConfig(
    # 数据规模相关
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=500,
    num_train_epochs=1,
    # 训练日志相关
    logging_strategy='steps',
    logging_steps=50,
    report_to='tensorboard',
    # 学习率和优化器
    learning_rate=3.0e-5,
    lr_scheduler_type='cosine',
    warmup_steps=0.1,
    # optim='adamw',
    # 评估和保存相关
    eval_strategy='steps',
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=2,
    output_dir='/root/tf-logs',
    # 优化相关
    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=True,
    max_length=700,
    # 其他参数
    assistant_only_loss=True,
    chat_template_path='test_chat_template.jinja'
)

# 2.2 构造SFTTrainer
sft_trainer = SFTTrainer(model=model,
                         processing_class=tokenizer,
                         train_dataset=dataset_dict['train'],
                         eval_dataset=dataset_dict['test'],
                         args=sft_config)


# dataloader = sft_trainer.get_train_dataloader()
# for batch in dataloader:
#     # print(batch)
#     print(tokenizer.decode(batch['input_ids']))
#     break


# 2.3 开始训练
sft_trainer.train()

# 2.4 保存模型
sft_trainer.save_model('model/Qwen3-0.6B-sft')

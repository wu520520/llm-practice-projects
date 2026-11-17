from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import os
# os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs"  # [修改] 注释掉原Linux路径,云服务上若需TensorBoard可改为相对路径如"./tf-logs"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # [新增] 解决显存碎片化导致的OOM

# 0 准备工作
model = AutoModelForCausalLM.from_pretrained('model/Qwen3-0.6B', torch_dtype='auto')  # [修改] 添加torch_dtype='auto',避免默认fp32占显存
tokenizer = AutoTokenizer.from_pretrained('model/Qwen3-0.6B')

# 打印模块名用于确认 target_modules 是否正确
print("=== 模型中的线性层模块 ===")
for name, module in model.named_modules():
    if 'proj' in name:
        print(name)

# 1.数据处理
# 1.1 加载数据
dataset_dict = load_dataset('json', data_files={'train': 'data/keywords_data_train.jsonl',
                                                'test': 'data/keywords_data_test.jsonl'})

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


# 2 lora微调
# 2.1 定义lora配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # [修改] 原lora_alpha=16,改为32,遵循alpha=2*r的经验法则,scaling更稳定
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # [修改] 原target_modules="all-linear",显存充足时显式列出7个线性层,效果更好且可控
    lora_dropout=0.1,  # [修改] 原lora_dropout=0.05,提高到0.1,数据量不大时缓解过拟合
    task_type="CAUSAL_LM",
)
# 2.2 根据lora配置给模型添加旁支
peft_model = get_peft_model(model, lora_config)


# 3 训练模型
# 3.1 配置类
sft_config = SFTConfig(
    # 数据规模相关 (有效batch = 16*2 = 32)
    per_device_train_batch_size=16,  # [修改] 原=8,seq砍半后显存充足,拉回16加速训练
    per_device_eval_batch_size=16,  # [修改] 原=8,同上
    gradient_accumulation_steps=2,  # [修改] 原=4,改成2保持有效batch=32,步数减半
    num_train_epochs=3,  # 完整跑3轮,效果优先
    # 训练日志相关
    logging_strategy='steps',
    logging_steps=10,
    report_to='tensorboard',
    # 学习率和优化器
    learning_rate=2.0e-4,
    lr_scheduler_type='cosine',
    warmup_steps=500,
    # 评估和保存相关
    eval_strategy='steps',
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=3,
    output_dir='./checkpoints',
    # 优化相关
    bf16=True,
    gradient_checkpointing=False,  # [修改] 原=True,关checkpointing大幅提速,seq=512后显存够
    max_length=512,  # [修改] 原=1024,砍半减显存,关键词任务文本不长,512足够覆盖
    # 其他参数
    assistant_only_loss=True,
    chat_template_path='test_chat_template.jinja',
    seed=42,
)

# 2.2 构造SFTTrainer
sft_trainer = SFTTrainer(model=peft_model,
                         processing_class=tokenizer,
                         train_dataset=dataset_dict['train'],
                         eval_dataset=dataset_dict['test'],
                         args=sft_config)




# 2.3 开始训练
sft_trainer.train()

# 2.4 保存模型
sft_trainer.save_model('model/Qwen3-0.6B-sft-lora')


"""
/root/miniconda3/envs/llm/lib/python3.11/site-packages/transformers/integrations
"""

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


import os
# 修改: 改为相对路径, 避免 Linux 服务器上 /root/tf-logs 不存在导致报错
os.environ["TENSORBOARD_LOGGING_DIR"] = "./tf-logs"

# 0 准备工作
model = AutoModelForCausalLM.from_pretrained('model/Qwen3-0.6B')
model.warnings_issued = {}
tokenizer = AutoTokenizer.from_pretrained('model/Qwen3-0.6B')



# 1 处理数据
# 修改: 数据从 1000 条扩到 5000 条 (总量 61135), 更多数据可显著减少过拟合
train_data = load_dataset('./data/ultrafeedback_binarized', split='train_prefs').select(range(5000))
test_data = load_dataset('./data/ultrafeedback_binarized', split='test_prefs')
train_data = train_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])
test_data = test_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])

# 2 创建DpoConfig
dpo_config = DPOConfig(
# 数据规模相关
    # 修改: DPO 需计算全词表 entropy (Qwen3 词表 ~151k), bs 过大会 OOM
    #       bs=2 + grad_accum=8, 有效 batch=16, 兼顾显存和效果
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    # 修改: 5000 条数据 × 2 epoch, 兼顾学习充分和防止过拟合
    num_train_epochs=2,
    # 训练日志相关
    logging_strategy='steps',
    logging_steps=10,
    report_to='tensorboard',
    # 学习率和优化器
    # 修改: DPO 对学习率敏感, 3e-4 偏高易导致训练不稳定, 降为 1e-4 更稳定
    learning_rate=1.0e-4,
    lr_scheduler_type='cosine',
    # 修改: 总步数约 625 (5000条×2epoch÷16), 10% 预热 ≈ 63 步
    warmup_steps=63,
    # 评估和保存相关
    eval_strategy='steps',
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=3,
    output_dir='./checkpoints',  # 保存检查点，日志等信息
    # 优化相关
    bf16=True,
    # 修改: 2500 偏长导致 OOM, 降为 1024 覆盖大部分样本且大幅省显存
    max_length=1024,
    # 修改: 新增 — 预计算参考模型 log_probs, 训练时无需再跑 ref 模型前向, 显著省显存
    precompute_ref_log_probs=True,
    # 额外参数
    beta=0.1
)

# 3 lora 配置
lora_config = LoraConfig(
    r=16,
    # 修改: lora_alpha 应为 2*r=32, 使 LoRA 输出缩放合理 (原 alpha=r=16 缩放不足)
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 4 添加lora旁支
peft_model = get_peft_model(model, lora_config)


# 5 构造DPOTrainer
dpo_trainer = DPOTrainer(
    model=peft_model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
)

# 6 开始训练
dpo_trainer.train()

# 7 保存模型 (LoRA adapter)
dpo_trainer.save_model('./model/Qwen3-0.6B-dpo')

# 修改: 新增合并步骤 — 将 LoRA adapter 合并进基座模型并保存完整模型
#       这样 predict.py 可直接用 AutoModelForCausalLM 加载, 无需单独加载 adapter
merged_model = dpo_trainer.model.merge_and_unload()
merged_model.save_pretrained('./model/Qwen3-0.6B-merged')
tokenizer.save_pretrained('./model/Qwen3-0.6B-merged')
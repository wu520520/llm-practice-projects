import time
import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import SftConfig
from data_process import get_train_data, create_answer_mask


def compute_loss(output_logits, target_labels, assistant_answer_mask):
    """损失函数
    output_logits.shape = (batch_size, seq_length, vocab_size)
    target_labels.shape = (batch_size, seq_length)
    assistant_answer_mask.shape = (batch_size, seq_length)
    """
    # 对output_logits进行softmax处理，得到每个token的概率
    log_probabilities = torch.log_softmax(output_logits, dim=-1)
    # 通过gather函数，得到model输出的target token处所对应的位置
    gathered_log_probs = torch.gather(log_probabilities, dim=-1, index=target_labels.unsqueeze(-1))  # gathered_props.shape = (batch_size, seq_length, 1)
    # 计算负对数似然损失
    negative_log_likelihood = gathered_log_probs*(-1)
    token_losses = negative_log_likelihood.squeeze(-1)  # gathered_props.shape = (batch_size, seq_length)
    # 对token_losses进行mask，只关注answer部分的loss
    masked_token_losses = torch.mul(token_losses, assistant_answer_mask)
    # 只对batch中所有有效token做平均，更接近SFTTrainer默认的token-level mean口径
    avg_loss = torch.sum(masked_token_losses) / torch.sum(assistant_answer_mask)
    return avg_loss

def cosine_decay(current_step, total_steps, warmup_ratio, max_lr):
    """学习率余弦衰减算法"""
    # 学习率预热步数
    warmup_steps = total_steps * warmup_ratio
    # 预热斜率K
    k = max_lr / warmup_steps
    if current_step <= warmup_steps:
        return k * current_step
    else:
        # 衰减进度，从1到0
        process = (current_step - warmup_steps) / (total_steps - warmup_steps)  # 进度，从0到1
        # 余弦衰减
        decay = (1 + np.cos(np.pi * process)) * 0.5  # [1, -1] --+1--> [2,0] --/2--> [1,0]
        return max_lr * decay


def train(model, config, tokenizer):
    """
    训练函数
    :param model: 模型
    :param config: 配置
    :param tokenizer: 分词器
    :return: 模型 分词器
    """
    # 模型放到设备(GPU)开启训练模式
    model.to(config.device)
    model.train()
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # 创建日志写入器
    writer = SummaryWriter(config.log_path)
    # 获取数据
    train_data = get_train_data(config, tokenizer)
    # one epoch(一轮训练) 获取总的训练步数
    total_steps = len(train_data) // config.batch_size
    # 开始一步步的训练
    step_losses = []
    # 创建进度条对象
    progress_bar = tqdm.tqdm(total=total_steps, desc="训练")
    for step in range(total_steps):
        # 获取一批数据
        batch_data = train_data[step * config.batch_size:(step + 1) * config.batch_size]

        # 对本批次数据进行padding
        # 获取pad_token_id
        pad_token_id = tokenizer.pad_token_id
        # 计算本批次数据的最大长度
        batch_max_len = max(len(data) for data in batch_data)
        padded_input_ids = []
        # 对每一条数据进行padding
        for data in batch_data:
            # 计算当前数据需要padding的长度
            padding_len = batch_max_len - len(data)
            # 进行padding
            padded_input_id = data + [pad_token_id] * padding_len
            padded_input_ids.append(padded_input_id)

        # 构建模型输入和目标值
        batch_data_tensor = torch.tensor(padded_input_ids)  # shape = (batch_size, seq_length)
        batch_inputs = batch_data_tensor[:, :-1].to(config.device)
        batch_labels = batch_data_tensor[:, 1:].to(config.device)

        # 计算模型输出的掩码
        padding_mask = torch.where(batch_labels == pad_token_id, 0, 1)  # 将padding的部分设为0，否则设为1
        assistant_answer_mask = create_answer_mask(batch_labels, tokenizer).to(config.device)
        final_loss_mask = torch.mul(padding_mask, assistant_answer_mask)

        # 前向传播
        outputs = model(batch_inputs).logits
        # 损失计算
        step_loss = compute_loss(outputs, batch_labels, final_loss_mask)
        step_losses.append(step_loss.item())
        # 反向传播
        step_loss.backward()
        # 使用余弦衰减更新学习率
        current_learning_rate = cosine_decay(step+1, total_steps, config.warmup_ratio, config.learning_rate)
        for x in optimizer.param_groups:
            x['lr'] = current_learning_rate
        writer.add_scalar("train_lr", current_learning_rate, step + 1)
        # 参数更新
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 控制进度条向前走一步
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{step_losses[-1]:.4f}", lr=f"{current_learning_rate:.2e}")

        # 每隔固定的步数写入训练日志
        if (step+1) % config.log_iter == 0 or (step+1) == total_steps:
            # 计算log_iter过程的平均损失
            current_iter_loss = step_losses[-config.log_iter:]
            current_iter_avg_loss = np.nanmean(current_iter_loss)
            writer.add_scalar('train_loss', current_iter_avg_loss, step+1)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"时间：{current_time} | "
                  f"步数：{step + 1}/{total_steps} | "
                  f"最近{len(current_iter_loss)}批次平均损失：{current_iter_avg_loss:.4f} | "
                  # f"当前学习率：{current_learning_rate:.2e}"
             )
    print("训练完成")
    return model, tokenizer


def save_model_tokenier(model, tokenizer, save_path):
    """保存模型和分词器"""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def main():
    config = SftConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    n_model, n_tokenier = train(model, config, tokenizer)
    save_model_tokenier(n_model, n_tokenier, config.save_path)


if __name__ == '__main__':
    # props = torch.tensor([[[0.1, 0.3, 0.5, 0.1], [0.1, 0.6, 0.2, 0.1]]])
    # targets = torch.tensor([[1, 2]])
    # assistant_answer_mask =torch.tensor([[0, 1]])
    # print(compute_loss(props, targets, assistant_answer_mask))
    # config = SftConfig()
    # tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # model = AutoModelForCausalLM.from_pretrained(config.model_path)
    # train(model, config, tokenizer)
    main()


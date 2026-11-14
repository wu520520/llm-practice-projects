import time
import tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import DpoConfig
from data_process import get_train_data, create_answer_mask


def compute_loss(chosen_log_props, reject_log_props, ref_chosen_log_props, ref_reject_log_props, beta):
    """
    计算一个批次平均损失
    :param chosen_log_props: 被训练的模型，输出喜欢答案的对数概率 shape = (batch_size,) chosen_log_props[0] 代表模型基于第0个样本的prompt 输出喜欢答案的概率
    :param reject_log_props: 被训练的模型，输出拒绝答案的对数概率 shape = (batch_size,) reject_log_props[0] 代表模型基于第0个样本的prompt 输出拒绝答案的概率
    :param ref_chosen_log_props: 参考模型，输出喜欢答案的对数概率 shape = (batch_size,) ref_chosen_log_props[0] 代表参考模型基于第0个样本的prompt 输出喜欢答案的概率
    :param ref_reject_log_props: 参考模型，输出拒绝答案的对数概率 shape = (batch_size,) ref_reject_log_props[0] 代表参考模型基于第0个样本的prompt 输出拒绝答案的概率
    :param beta: 超参数，控制偏好的强度
    :return: 本批次的平均损失
    """
    margin = chosen_log_props - reject_log_props - (ref_chosen_log_props - ref_reject_log_props)
    result = (-1) * torch.nn.functional.logsigmoid(margin * beta)
    return result.sum() / len(result)


def compute_log_loss(output_logits, labels, assistant_answer_mask):
    """损失函数
    output_logits.shape = (batch_size, seq_length, vocab_size)
    labels.shape = (batch_size, seq_length)
    assistant_answer_mask.shape = (batch_size, seq_length)
    """
    # 对output_logits进行softmax处理，得到每个token的概率
    log_probs = torch.log_softmax(output_logits, dim=-1)
    # 通过gather函数，得到model输出的target token处所对应的位置
    labels_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))  # gathered_props.shape = (batch_size, seq_length, 1)
    # 计算负对数似然损失
    negative_log_likelihood = labels_log_probs*(-1)
    token_losses = negative_log_likelihood.squeeze(-1)  # gathered_props.shape = (batch_size, seq_length)
    # 对token_losses进行mask，只关注answer部分的loss
    masked_token_losses = torch.mul(token_losses, assistant_answer_mask)
    # 通过.sum传入dim=-1,计算每一个样本模型输出的回答的对数概率
    avg_loss = torch.sum(masked_token_losses, dim=-1) / torch.sum(assistant_answer_mask, dim=-1)
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


def train(config):
    """
    训练函数
    :param config: 配置
    :return: 模型 分词器
    """
    # 0.  ======================== 准备阶段 ============================
    # 0.1 加载模型
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)  # 要训练的模型
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_path) # 参考模型
    # 0.2 模型放到设备(GPU)开启训练模式
    model.to(config.device)
    model.train()
    ref_model.to(config.device)
    ref_model.eval()
    # 0.3 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # 0.4 创建日志写入器
    writer = SummaryWriter(config.log_path)
    # 0.5 获取数据
    chosen_result_list, reject_result_list = get_train_data(config, tokenizer)
    # 0.6 获取总的训练步数
    total_steps = len(chosen_result_list) // config.batch_size


    # 1.  ======================== 训练阶段 ============================
    # 记录损失
    step_losses = []
    # 创建进度条对象
    progress_bar = tqdm.tqdm(total=total_steps, desc="训练")
    for step in range(total_steps):
        # 1.1  构造数据

        # 1.1.1 构造训练模型的数据(chosen) chosen_input_ids, chosen_labels, chosen_assistant_answer_mask
        # 获取一批数据
        chosen_batch_train_data = chosen_result_list[step * config.batch_size:(step + 1) * config.batch_size]
        # 获取该批次数据的最大长度
        chosen_max_batch_len = max(len(data) for data in chosen_batch_train_data)
        # 对该批数据进行padding
        for data in chosen_batch_train_data:
            # 计算当前数据需要padding的长度
            padding_len = chosen_max_batch_len - len(data)
            # 进行padding
            data.extend([tokenizer.pad_token_id] * padding_len)
        # 转换为张量
        chosen_data_tensor = torch.tensor(chosen_batch_train_data, dtype=torch.long).to(config.device)
        # 构建输入和目标数据
        chosen_input_ids = chosen_data_tensor[:, :-1]
        chosen_labels = chosen_data_tensor[:, 1:]
        # 构建答案的掩码
        chosen_assistant_answer_mask = create_answer_mask(chosen_labels, tokenizer)

        # 1.1.2 构造训练模型的数据(reject) reject_input_ids, reject_labels, reject_assistant_answer_mask
        # 获取一批数据
        reject_batch_train_data = reject_result_list[step * config.batch_size:(step + 1) * config.batch_size]
        # 获取该批次数据的最大长度
        reject_max_batch_len = max(len(data) for data in reject_batch_train_data)
        # 对该批数据进行padding
        for data in reject_batch_train_data:
            # 计算当前数据需要padding的长度
            padding_len = reject_max_batch_len - len(data)
            # 进行padding
            data.extend([tokenizer.pad_token_id] * padding_len)
        # 转换为张量
        reject_data_tensor = torch.tensor(reject_batch_train_data, dtype=torch.long).to(config.device)
        # 构建输入和目标数据
        reject_input_ids = reject_data_tensor[:, :-1]
        reject_labels = reject_data_tensor[:, 1:]
        # 构建答案的掩码
        reject_assistant_answer_mask = create_answer_mask(reject_labels, tokenizer)



        # 1.2 前向传播
        # 1.2.1 训练模型的前向传播--> 基于chosen和reject的数据进行两次前向传播
        chosen_output_logits = model(chosen_input_ids).logits
        reject_output_logits = model(reject_input_ids).logits

        # 1.2.2 参考模型的前向传播--> 基于chosen和reject的数据进行两次前向传播
        with torch.no_grad():
            ref_chosen_output_logits = ref_model(chosen_input_ids).logits
            ref_reject_output_logits = ref_model(reject_input_ids).logits

        # 1.3 损失计算
        # 1.3.1 训练模型前向传播的结果
        chosen_log_props = compute_log_loss(chosen_output_logits, chosen_labels, chosen_assistant_answer_mask)
        reject_log_props = compute_log_loss(reject_output_logits, reject_labels, reject_assistant_answer_mask)
        # 1.3.2 参考模型前向传播的结果
        ref_chosen_log_props = compute_log_loss(ref_chosen_output_logits, chosen_labels, chosen_assistant_answer_mask)
        ref_reject_log_props = compute_log_loss(ref_reject_output_logits, reject_labels, reject_assistant_answer_mask)
        # 1.3.3 计算损失
        loss = compute_loss(chosen_log_props, reject_log_props, ref_chosen_log_props, ref_reject_log_props, config.beta)
        step_losses.append(loss.item())

        # 1.4 反向传播
        loss.backward()
        # 1.5 使用余弦衰减更新学习率
        current_learning_rate = cosine_decay(step+1, total_steps, config.warmup_ratio, config.learning_rate)
        for x in optimizer.param_groups:
            x['lr'] = current_learning_rate
        writer.add_scalar("train_lr", current_learning_rate, step + 1)
        # 1.6 参数更新
        optimizer.step()
        # 1.7 梯度清零
        optimizer.zero_grad()

        # 1.8 控制进度条向前走一步
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{step_losses[-1]:.4f}", lr=f"{current_learning_rate:.2e}")

        # 1.9 每隔固定的步数写入训练日志
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
    config = DpoConfig()
    model, tokenier = train(config)
    save_model_tokenier(model, tokenier, config.save_path)


if __name__ == '__main__':
    main()


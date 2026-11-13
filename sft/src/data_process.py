"""数据处理"""
import datasets
import torch
from transformers import AutoTokenizer
from config import SftConfig


def get_train_data(config, tokenizer):
    # 1.加载训练数据集
    train_dataset = datasets.load_dataset(str(config.data_path))['train']
    result_list = []
    # 2.遍历并转换数据格式
    for i in range(config.train_data_size):
        # 获取一条数据
        message_list = train_dataset[i]['messages']
        message_list.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        # 使用apply_chat_template进行转换
        result = tokenizer.apply_chat_template(message_list, tokenize=True, truncate=True, max_length=2000)
        result_list.append(result)
    # 3.返回
    return result_list



def create_answer_mask(input_ids, tokenizer):
    """
      输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

      以下面的对话为例：
      <|im_start|>system
      You are a helpful assistant.<|im_end|>
      <|im_start|>user
      什么是习惯？<|im_end|>
      <|im_start|>assistant
      习惯是指在一定时间内重复执行的行为。<|im_end|>
      <|im_start|>user
      如何培养一个习惯<|im_end|>
      <|im_start|>assistant
      21天培养法，每天坚持xxx<|im_end|>

      假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
      那么输入的eos_token_id为：[10,15,20,25]
      user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

      输出结果为：
                user_turns:[11,21]
                assistant_ends:[16,26]

    """
    # 1.创建和input_ids 形状相同且全部为0的张量
    answer_mask = torch.zeros_like(input_ids)  # input_ids.shape=[[1,2,3,4,5...],[1,2,3,4,5...],[1,2,3,4,5...],...]
    # 2.遍历每一条数据
    for idx, ids in enumerate(input_ids):
        eos_token_id = tokenizer.encode('<|im_end|>')[0]
        eos_positions = torch.where(ids == eos_token_id)[0].tolist()  # [5,10,15,20,25]
        eos_positions = eos_positions[1:]  # [10,15,20,25]
        # 区分哪个是user的结束符 哪个是AI的结束符
        user_ends, assistant_ends = _parse_conversation_turns(eos_positions)
        # 找出答案的掩码位置
        _set_answer_masks(answer_mask[idx], user_ends, assistant_ends)
    return answer_mask


def _parse_conversation_turns(eos_position):  # [10,15,20,25]
    # 从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符） 每隔一个取一次
    user_ends = [pos+1 for pos in eos_position[::2]]
    # assistant_turns为从第二个开始取，每隔一个取一次。
    assistant_ends = [pos+1 for pos in eos_position[1::2]]
    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends):
    # 判断用户和AI的轮数 user_ends[11,21] assistant[16,26]
    user_turns, assistant_turns = len(user_ends), len(assistant_ends)

    # 用户和AI轮数相同
    if user_turns == assistant_turns:
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
    # 用户和AI轮数不相同
    if user_turns == assistant_turns + 1:
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

        # 处理被截断的AI答案
        answer_start = user_ends[-1] + 3
        mask[answer_start:] = 1



if __name__ == '__main__':
    config = SftConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # result = get_train_data(config, tokenizer)
    # print(result[0], len(result))
    # print(tokenizer.decode(tokenizer.eos_token_id))

    """
    data_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好？"},
        {"role": "assistant", "content": "你也好"},
        {"role": "user", "content": "今天天气怎么样"},
        {"role": "assistant", "content": "今天天气很好"},
    ]

    print(tokenizer.apply_chat_template(data_list, tokenize=False))
    input_ids = tokenizer.apply_chat_template(data_list, tokenize=True)
    labels = torch.tensor(input_ids[:]).unsqueeze(0)
    answer_mask = create_answer_mask(labels, tokenizer)[0].tolist()
    print(answer_mask)

    for label_token, mask in zip(labels.tolist()[0],answer_mask):
        print(f"当前的token是ID{repr(label_token)}, 对应的token是{repr(tokenizer.decode(label_token))},对应的mask是{mask}")
        
    """

    # train_list = get_train_data(config, tokenizer)
    # print(len(train_list))
    # sample_data = train_list[0:1]
    # print(sample_data)
    # sample_data_tensor = torch.tensor(sample_data)
    # answer_mask = create_answer_mask(sample_data_tensor, tokenizer)
    # print(answer_mask.tolist())


    # ---------------- 如何统计数据的长度 ----------------
    train_data = get_train_data(config, tokenizer)
    data_lengths = [len(data) for data in train_data]
    import pandas as pd
    df = pd.DataFrame(data_lengths, columns=['data_length'])
    # print(df.describe())
    print(df['data_length'].quantile(0.95))
    print(df['data_length'].quantile(0.96))
    print(df['data_length'].quantile(0.97))
    print(df['data_length'].quantile(0.98))
    print(df['data_length'].quantile(0.99))
    print(df['data_length'].quantile(0.999))
    print(df['data_length'].quantile(0.9999))

    # for data in train_data:
    #     print(data)

    # print(tokenizer.pad_token_id)







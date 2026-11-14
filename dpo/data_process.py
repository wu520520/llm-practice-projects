"""数据处理"""
import datasets
import torch
from transformers import AutoTokenizer
from config import DpoConfig


def get_train_data(config, tokenizer):
    # 1.加载训练数据集
    train_dataset = datasets.load_dataset(str(config.data_path))['train_prefs']
    chosen_result_list = []
    reject_result_list = []
    # 2.遍历并转换数据格式
    for i in range(config.train_data_size):
        # 获取一条数据
        chosen_list = train_dataset[i]['chosen']
        reject_list = train_dataset[i]['rejected']
        # 使用apply_chat_template进行转换
        chosen_result = tokenizer.apply_chat_template(chosen_list, tokenize=True, truncate=True, max_length=2000)
        chosen_result_list.append(chosen_result)
        reject_result = tokenizer.apply_chat_template(reject_list, tokenize=True, truncate=True, max_length=2000)
        reject_result_list.append(reject_result)
    # 3.返回
    return chosen_result_list,reject_result_list



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
    config = DpoConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    chosen_result_list, reject_result_list = get_train_data(config, tokenizer)
    print(len(chosen_result_list))
    print(chosen_result_list[0], tokenizer.decode(chosen_result_list[0]))
    print(reject_result_list[0], tokenizer.decode(reject_result_list[0]))








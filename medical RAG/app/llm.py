"""本地 Qwen3 生成模型。PyTorch + Transformers，单例，懒加载。"""
from __future__ import annotations

import re
import threading

import torch

from app import config

# 匹配 Qwen3 思考标签：成对的 <think>...</think> 和孤立标签
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.S)
_THINK_TAG_SINGLE_RE = re.compile(r"</?think>")


def strip_think_tags(text: str) -> str:
    """清除思考标签，避免输出推理过程污染答案。"""
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = _THINK_TAG_SINGLE_RE.sub("", cleaned)
    return cleaned.strip()


class LocalLLM:
    """本地 Qwen3-0.6B 对话模型（单例，第一次调用 generate 时才加载）。"""

    _instance: "LocalLLM | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> "LocalLLM":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str | None = None) -> None:
        if getattr(self, "_model", None) is not None:
            return  # 单例已初始化过
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # GPU → bfloat16 加速，CPU → float32 兼容
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_path = model_path or config.LLM_MODEL_PATH

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """消息列表 → 生成答案。"""
        max_new_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature

        # 用模型自带的 ChatML 模板把 messages 拼成 prompt
        # enable_thinking=False：关闭 Qwen3 默认的推理过程输出，直接给答案
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 旧版 tokenizer 不支持 enable_thinking，降级处理
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # 贪心解码（温度 0）+ repetition_penalty 缓解小模型重复
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 只截取新生成的部分（跳过输入）
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_think_tags(text)

"""文本嵌入模型（BGE-M3，输出 1024 维稠密向量）。"""
from __future__ import annotations

import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app import config


class BGEEmbedder:
    """BGE-M3 嵌入器（线程安全单例，进程内只加载一次模型）。"""

    _instance: "BGEEmbedder | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> "BGEEmbedder":
        # 双重检查锁定：只创建一次实例，保证 FastAPI 并发安全
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str | None = None) -> None:
        if getattr(self, "_model", None) is not None:
            return  # 单例已初始化过，直接跳过
        # 有 CUDA 用 GPU，否则 CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or config.EMBED_MODEL_PATH
        self._model = SentenceTransformer(self.model_path, device=self.device)
        self.dim = self._model.get_sentence_embedding_dimension()  # BGE-M3 固定 1024

    def encode(
        self,
        texts: str | list[str],
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """文本 → float32 向量。normalize=True 时做 L2 归一化（余弦检索必需）。"""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size or config.EMBED_BATCH_SIZE,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings[0] if single else embeddings


def get_embedder() -> BGEEmbedder:
    """获取嵌入器单例。"""
    return BGEEmbedder()

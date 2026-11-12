"""FAISS 向量索引：L2 归一化 + IndexFlatIP = 余弦相似度 Top-K 检索。"""
from __future__ import annotations

import json
from typing import Sequence

import faiss
import numpy as np
from langchain_core.documents import Document

from app import config
from app.embeddings import BGEEmbedder, get_embedder


class FAISSVectorStore:
    """封装 FAISS 索引与文档列表。FAISS 只存向量，文档单独存 JSON。"""

    def __init__(self, index: faiss.Index, documents: list[Document]) -> None:
        self.index = index
        self.documents = documents  # 与向量行号一一对应

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        embedder: BGEEmbedder | None = None,
    ) -> "FAISSVectorStore":
        """批量编码文档 + 构建 IndexFlatIP 索引。"""
        embedder = embedder or get_embedder()
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.encode(texts)                 # (n, 1024)，已 L2 归一化
        embeddings = cls._l2_normalize(embeddings)          # 兜底再归一化一次

        # IndexFlatIP = 精确暴力 + 内积；向量归一化后 内积=余弦相似度
        index = faiss.IndexFlatIP(embeddings.shape[1])      # 1024 维
        index.add(embeddings)
        return cls(index=index, documents=list(documents))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[tuple[Document, float]]:
        """查询向量 → Top-K (文档, 余弦相似度)。"""
        # 把查询向量 reshape 成 (1, dim) 并归一化
        query = self._l2_normalize(
            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        )
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)  # FAISS 返回 (分数数组, 行号数组)

        results: list[tuple[Document, float]] = []
        for rank, doc_idx in enumerate(indices[0]):
            if doc_idx < 0:
                continue  # 结果不足 top_k 时 FAISS 返回 -1 占位
            results.append((self.documents[int(doc_idx)], float(scores[0][rank])))
        return results

    def save(
        self,
        index_path: str | None = None,
        docs_path: str | None = None,
    ) -> None:
        """持久化到 storage/。

        Windows 下 FAISS 原生 write_index 不支持中文路径，
        所以用 serialize_index → numpy → Python open 绕过。
        """
        config.ensure_dirs()
        index_path = index_path or str(config.FAISS_INDEX_PATH)
        docs_path = docs_path or str(config.DOCS_META_PATH)

        # FAISS → uint8 数组 → 写入二进制文件
        index_array = np.asarray(faiss.serialize_index(self.index), dtype="uint8")
        with open(index_path, "wb") as f:
            f.write(index_array.tobytes())

        # 文档 → JSON（含 page_content 和 metadata）
        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_path: str | None = None,
        docs_path: str | None = None,
    ) -> "FAISSVectorStore":
        """从磁盘加载。"""
        index_path = index_path or str(config.FAISS_INDEX_PATH)
        docs_path = docs_path or str(config.DOCS_META_PATH)

        # 读二进制 → uint8 数组 → 反序列化为 FAISS 索引
        with open(index_path, "rb") as f:
            index_array = np.frombuffer(f.read(), dtype="uint8")
        index = faiss.deserialize_index(index_array)

        with open(docs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload
        ]
        return cls(index=index, documents=documents)

    @staticmethod
    def exists() -> bool:
        """判断持久化文件是否都在。"""
        return config.FAISS_INDEX_PATH.exists() and config.DOCS_META_PATH.exists()

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 归一化：v / ||v||。分母加 clip 防止除零。"""
        return vectors / np.clip(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None
        )

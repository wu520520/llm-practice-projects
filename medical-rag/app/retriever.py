"""语义检索：用户问题 → BGE-M3 编码 → FAISS Top-K 余弦检索。"""
from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.data_loader import load_and_clean, build_documents
from app.embeddings import BGEEmbedder, get_embedder
from app.vector_store import FAISSVectorStore


@dataclass
class RetrievedDoc:
    """单条检索结果，供 service / API 层使用。"""
    title: str            # 疾病名称
    content: str          # 结构化疾病知识全文
    score: float          # 余弦相似度（-1 ~ 1，越大越相关）
    metadata: dict        # 完整元数据（症状/药物/科室/标准答案等）


class MedicalRetriever:
    """医疗知识库 Top-K 检索器。"""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder or get_embedder()

    @classmethod
    def from_storage(cls) -> "MedicalRetriever":
        """从 storage/ 加载索引；没有的话自动清洗数据 + 建库。"""
        if FAISSVectorStore.exists():
            store = FAISSVectorStore.load()
        else:
            # 首次运行：load_and_clean → build_documents → 向量化 → save
            dataframe = load_and_clean()
            documents = build_documents(dataframe)
            store = FAISSVectorStore.build(documents)
            store.save()
        return cls(store)

    def retrieve(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[RetrievedDoc]:
        """问题 → 编码 → FAISS 检索 → 封装为 RetrievedDoc 列表。"""
        query_vector = self.embedder.encode(question.strip())
        hits = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievedDoc(
                title=doc.metadata.get("disease", "未知疾病"),
                content=doc.page_content,
                score=score,
                metadata=doc.metadata,
            )
            for doc, score in hits
        ]


def format_context(docs: list[RetrievedDoc]) -> str:
    """把多篇文档拼成生成模型的上下文：带编号、来源、相似度。"""
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        blocks.append(
            f"【资料 {idx}】（来源：{doc.title}，相似度：{doc.score:.4f}）\n"
            f"{doc.content}"
        )
    return "\n\n" + ("=" * 20) + "\n\n".join(blocks)

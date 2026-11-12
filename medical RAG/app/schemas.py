"""任务三：FastAPI 请求 / 响应 Pydantic 模型与示例。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """POST /qa/ask 请求体。"""

    question: str = Field(
        ..., min_length=1, description="用户的医疗健康问题",
        json_schema_extra={"example": "糖尿病有哪些典型症状？"},
    )
    session_id: str = Field(
        default="default",
        description="会话 ID，相同 ID 的请求共享多轮对话短期记忆",
    )
    top_k: int = Field(
        default=3, ge=1, le=20, description="检索的相关文档数量",
    )
    use_history: bool = Field(
        default=True, description="是否携带该会话的历史对话参与生成",
    )


class ReferenceDoc(BaseModel):
    """单条检索参考文档。"""

    title: str = Field(..., description="文档标题（疾病名称）")
    content: str = Field(..., description="文档全文")
    score: float = Field(..., description="余弦相似度分数")
    department: str = Field("", description="推荐就诊科室")


class AskResponse(BaseModel):
    """POST /qa/ask 响应体。"""

    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="RAG 生成的最终答案")
    sources: list[str] = Field(
        ..., description="参考来源（Top-3 文档标题 / 疾病名称）",
    )
    references: list[ReferenceDoc] = Field(
        ..., description="检索到的 Top-K 文档详情",
    )
    session_id: str = Field(..., description="会话 ID")


class RetrieveRequest(BaseModel):
    """POST /qa/retrieve 请求体（仅检索，不生成答案）。"""

    query: str = Field(
        ..., min_length=1, description="检索查询语句",
        json_schema_extra={"example": "头痛头晕可能是什么病？"},
    )
    top_k: int = Field(default=3, ge=1, le=20, description="返回文档数量")


class RetrieveResponse(BaseModel):
    """POST /qa/retrieve 响应体。"""

    query: str = Field(..., description="原始查询")
    total: int = Field(..., description="返回文档数量")
    references: list[ReferenceDoc] = Field(..., description="Top-K 文档列表")


class HealthResponse(BaseModel):
    """GET /qa/health 响应体。"""

    status: str = Field(..., description="服务状态")
    docs_in_index: int = Field(..., description="知识库文档总数")

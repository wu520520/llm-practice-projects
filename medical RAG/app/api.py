"""任务三：FastAPI 应用与路由定义。

路由：
    POST /qa/ask      —— 完整 RAG 问答，返回答案与 Top-3 参考来源
    POST /qa/retrieve —— 仅返回 Top-K 检索文档，供调试
    GET  /qa/health   —— 健康检查

启动后自动生成交互式 Swagger 文档：http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app import config
from app.schemas import (
    AskRequest, AskResponse, HealthResponse,
    ReferenceDoc, RetrieveRequest, RetrieveResponse,
)
from app.service import QAService

#: Swagger 文档元数据
app = FastAPI(
    title="医疗知识库 RAG 智能问答系统",
    description=(
        "基于医疗知识图谱与检索增强生成（RAG）技术的医疗健康问答服务。\n\n"
        "- 嵌入模型：BGE-M3（1024 维）\n"
        "- 向量索引：FAISS 余弦相似度 Top-K 检索\n"
        "- 生成模型：本地 Qwen3-0.6B\n"
        "- 支持基于短期记忆的多轮对话"
    ),
    version="1.0.0",
)

#: 全局 service 单例（检索器在启动时完成加载/建库，LLM 首次问答时懒加载）
_service: QAService | None = None


def get_service() -> QAService:
    """获取全局 QAService 单例。"""
    global _service
    if _service is None:
        _service = QAService()
    return _service


@app.on_event("startup")
def _startup() -> None:
    """服务启动时加载检索器（索引缺失会自动完成数据清洗与建库）。"""
    get_service()


@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    """根路径自动跳转 Swagger 文档。"""
    return RedirectResponse(url="/docs")


@app.get("/qa/health", response_model=HealthResponse, tags=["问答服务"],
         summary="健康检查")
def health() -> HealthResponse:
    """检查服务是否存活，并返回知识库文档数与生成后端模式。"""
    service = get_service()
    return HealthResponse(
        status="ok",
        docs_in_index=service.retriever.vector_store.index.ntotal,
    )


@app.post("/qa/retrieve", response_model=RetrieveResponse, tags=["问答服务"],
          summary="Top-K 文档检索（调试用，不生成答案）")
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """仅执行语义检索，返回最相关的 Top-K 疾病文档。"""
    try:
        docs = get_service().retrieve(request.query, top_k=request.top_k)
    except Exception as exc:  # 检索失败返回可读错误
        raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from exc

    return RetrieveResponse(
        query=request.query,
        total=len(docs),
        references=[
            ReferenceDoc(
                title=doc.title,
                content=doc.content,
                score=round(doc.score, 4),
                department=doc.metadata.get("department", ""),
            )
            for doc in docs
        ],
    )


@app.post("/qa/ask", response_model=AskResponse, tags=["问答服务"],
          summary="RAG 智能问答（支持多轮对话）")
def ask(request: AskRequest) -> AskResponse:
    """检索 Top-K 疾病资料，结合对话历史生成最终答案并给出参考来源。"""
    try:
        result = get_service().answer(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            use_history=request.use_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"答案生成失败：{exc}") from exc

    return AskResponse(**result)

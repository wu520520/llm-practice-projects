import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from .config import get_settings
from .schemas import ChatRequest

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one reusable async client for the lifetime of the API process."""
    app.state.llm_client = AsyncOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=1,
    )
    try:
        yield
    finally:
        await app.state.llm_client.close()


app = FastAPI(
    title="vLLM + FastAPI Inference Demo",
    description="A small FastAPI gateway calling vLLM through its OpenAI-compatible API.",
    version="1.0.0",
    lifespan=lifespan,
)


def _messages_payload(request: ChatRequest) -> list[dict[str, str]]:
    return [message.model_dump() for message in request.messages]


def _raise_upstream_error(exc: Exception) -> HTTPException:
    if isinstance(exc, APIConnectionError):
        return HTTPException(
            status_code=503,
            detail="Cannot connect to the vLLM server. Check VLLM_BASE_URL and whether vLLM is ready.",
        )
    if isinstance(exc, APIStatusError):
        return HTTPException(
            status_code=502,
            detail=f"vLLM returned HTTP {exc.status_code}: {exc.message}",
        )
    return HTTPException(status_code=502, detail=f"Inference request failed: {exc}")


@app.get("/health/live", tags=["health"])
async def liveness() -> dict[str, str]:
    """Only checks whether the FastAPI process is running."""
    return {"status": "alive"}


@app.get("/health", tags=["health"])
async def readiness() -> dict[str, str]:
    """Checks whether the upstream vLLM API can be reached."""
    try:
        await app.state.llm_client.models.list()
    except OpenAIError as exc:
        raise _raise_upstream_error(exc) from exc
    return {"status": "ready", "model": settings.served_model_name}


@app.post("/api/chat", tags=["inference"])
async def chat(request: ChatRequest) -> dict:
    """Non-streaming chat completion via the vLLM OpenAI-compatible endpoint."""
    try:
        result = await app.state.llm_client.chat.completions.create(
            model=settings.served_model_name,
            messages=_messages_payload(request),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except OpenAIError as exc:
        raise _raise_upstream_error(exc) from exc

    choice = result.choices[0]
    return {
        "id": result.id,
        "model": result.model,
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "usage": result.usage.model_dump() if result.usage else None,
    }


@app.post("/api/chat/stream", tags=["inference"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events (SSE): each data event contains one generated text delta."""
    try:
        stream = await app.state.llm_client.chat.completions.create(
            model=settings.served_model_name,
            messages=_messages_payload(request),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=True,
        )
    except OpenAIError as exc:
        raise _raise_upstream_error(exc) from exc

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    payload = json.dumps({"delta": delta}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # The HTTP response has started; report stream errors as SSE.
            payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

"""服务启动脚本。"""
from __future__ import annotations

import argparse

import uvicorn

from app import config


def main() -> None:
    parser = argparse.ArgumentParser(description="医疗知识库 RAG 问答系统")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--embedding-model", default=config.EMBED_MODEL_PATH)
    parser.add_argument("--llm-model", default=config.LLM_MODEL_PATH)
    parser.add_argument("--rebuild", action="store_true",
                        help="启动前重建 FAISS 索引")
    args = parser.parse_args()

    config.EMBED_MODEL_PATH = args.embedding_model
    config.LLM_MODEL_PATH = args.llm_model
    config.ensure_dirs()

    if args.rebuild:
        from app.data_loader import build_documents, load_and_clean
        from app.embeddings import get_embedder
        from app.vector_store import FAISSVectorStore
        print("[启动] 重建索引 ...")
        store = FAISSVectorStore.build(build_documents(load_and_clean()),
                                       embedder=get_embedder())
        store.save()
        print(f"[启动] 完成：{store.index.ntotal} 个文档向量")

    print(f"服务启动：http://localhost:{args.port}  Swagger: http://localhost:{args.port}/docs")
    uvicorn.run("app.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

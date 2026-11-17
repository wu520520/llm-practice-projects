"""任务一入口：数据清洗 + FAISS 索引构建。"""
from __future__ import annotations

import json
import time

from app import config
from app.data_loader import basic_stats, build_documents, load_and_clean, load_raw_data
from app.embeddings import get_embedder
from app.vector_store import FAISSVectorStore


def main() -> None:
    config.ensure_dirs()

    # 1. 加载 & 统计
    raw = load_raw_data()
    print(f"[1/4] 加载原始数据：{len(raw)} 条")

    # 2. 清洗
    dataframe = load_and_clean()
    removed_empty = dataframe.attrs.get("removed_empty", 0)
    removed_dup = dataframe.attrs.get("removed_dup", 0)
    print(f"[2/4] 清洗完成：删除空值 {removed_empty} 条、重复 {removed_dup} 条，"
          f"剩余 {len(dataframe)} 条，{dataframe['disease'].nunique()} 种疾病")

    # 保存统计
    stats = basic_stats(dataframe)
    stats_path = config.OUTPUT_DIR / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 3. 文档构建
    documents = build_documents(dataframe)
    print(f"[3/4] 构建结构化文档 {len(documents)} 篇")

    # 4. 向量化 & 建库
    embedder = get_embedder()
    print(f"[4/4] 编码 + 建库（{embedder.dim} 维，设备 {embedder.device}）...")
    t0 = time.time()
    store = FAISSVectorStore.build(documents, embedder=embedder)
    store.save()
    print(f"      完成：{store.index.ntotal} 个向量，耗时 {time.time() - t0:.2f}s")

    # 自检
    print("\n检索自检：糖尿病 → ", end="")
    top1 = store.search(embedder.encode("糖尿病有哪些典型症状？"), top_k=1)[0]
    print(f"[{top1[1]:.4f}] {top1[0].metadata.get('disease')}")


if __name__ == "__main__":
    main()

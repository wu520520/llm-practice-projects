"""医疗知识图谱 RAG 问答系统应用包。

模块划分：
    config        —— 全局配置（路径 / 模型 / 检索参数）
    data_loader   —— 任务一：医疗问答数据加载、清洗与结构化文档构建
    embeddings    —— 任务一：中文文本嵌入模型（BGE-M3，基于 PyTorch）
    vector_store  —— 任务一：FAISS 余弦相似度向量索引
    retriever     —— 任务二：Top-K 语义检索模块
    llm           —— 任务二：答案生成模型（本地 Qwen3-0.6B）
    memory        —— 任务二：多轮对话短期记忆
    service       —— 任务二：检索 → 生成完整业务编排（service 层）
    schemas       —— 任务三：FastAPI 请求 / 响应数据模型
    api           —— 任务三：FastAPI 路由
"""

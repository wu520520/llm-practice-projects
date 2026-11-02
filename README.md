# LLM 大模型应用与微调项目集

面向 **大模型应用工程师** 的实战项目集，覆盖 **RAG 检索增强生成 · Agent 智能体 · LangChain/LangGraph · LLM 微调（SFT / DPO / LoRA / QLoRA / TRL / Unsloth）** 全链路。

> 模型权重与大规模训练数据集体积过大，不入库；保留全部源码与演示数据，可本地下载权重后运行。

---

## 📂 项目构成

### 🤖 RAG 检索增强生成
- **`rag/`** — RAG 全流程：多格式文档加载（Markdown / Word / PDF）、文本切分、BGE 稠密+稀疏向量、Milvus 向量库建库/增删/检索
- **`medical RAG/`** — 医疗问答 RAG 完整应用：向量库 + FAISS 索引 + LLM 生成 + 评估报告（含 API 服务）

### 🧠 Agent 智能体
- **`agent/`** — OpenAI SDK / LangChain 工具调用、ReAct Agent、MCP Server/Client（stdio、streamable http）、Tavily 搜索

### 🔗 LangChain / LangGraph
- **`langchain/`** — 模型调用（OpenAI / Ollama / DeepSeek）、提示模板、输出解析、Runnable 链、并行、批处理/流式
- **`langgraph/`** — 状态图构建、Reducer、节点/输入输出隔离、记忆、Agent 中间件、MCP 集成、12306 Agent

### 🎯 LLM 微调（PEFT + 对齐）
- **`sft/`** — Qwen3-0.6B 监督微调（SFT），含数据处理与训练
- **`dpo/`** — DPO 偏好优化
- **`LoRA/` / `qlora/`** — LoRA 与 QLoRA 低秩微调（训练 / 合并 / 预测）
- **`loRA_dpo/`** — LoRA + DPO 组合微调
- **`trl/`** — 基于 TRL 的微调
- **`undloth/`** — Unsloth 快速微调

---

## 🛠 Tech Stack

Python · PyTorch · HuggingFace Transformers / TRL · PEFT · LangChain · LangGraph · OpenAI SDK · MCP · Milvus · FAISS · BGE · Qwen · vLLM · Unsloth

---

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 进入对应项目目录运行，例如 RAG 应用
cd "medical RAG"
python run.py
```

各子项目含独立代码与演示数据，模型权重请按需下载后放入对应 `model/` 目录。

---

## 📌 说明

- 模型权重（`*.safetensors` / `*.bin` / `*.onnx` 等）与训练数据集（`*.parquet` 及大 jsonl）已通过 `.gitignore` 排除，不入库
- 各子项目按主题独立组织，便于按需扩展与维护

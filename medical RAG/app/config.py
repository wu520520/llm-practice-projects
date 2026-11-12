"""全局配置模块。

所有路径与运行参数集中在此处管理，优先读取环境变量（或项目根目录下的
``.env`` 文件），便于在不修改代码的情况下切换模型路径、生成模式与服务端口。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 项目根目录（本文件位于 <项目根>/app/config.py）
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# 加载 .env（若存在），其中的变量不会覆盖系统已存在的环境变量
load_dotenv(BASE_DIR / ".env")

#: 原始医疗问答数据路径
DATA_PATH: Path = BASE_DIR / "medical_qa.csv"

#: FAISS 索引与文档元数据持久化目录
STORAGE_DIR: Path = BASE_DIR / "storage"

#: 索引文件路径（FAISS 原生二进制格式）
FAISS_INDEX_PATH: Path = STORAGE_DIR / "medical_faiss.index"

#: 文档内容与元数据持久化路径（与索引向量按行号一一对应）
DOCS_META_PATH: Path = STORAGE_DIR / "medical_docs.json"

#: 统计、评估等运行产物输出目录
OUTPUT_DIR: Path = BASE_DIR / "outputs"

# ---------------------------------------------------------------------------
# 嵌入模型配置（任务一）
# ---------------------------------------------------------------------------
#: 本地中文嵌入模型路径（BGE-M3，输出 1024 维稠密向量）
EMBED_MODEL_PATH: str = os.getenv(
    "EMBED_MODEL_PATH",
    "./rag/model/bge-m3",
)

#: 嵌入向量维度（BGE-M3 为 1024）
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1024"))

#: 编码批大小
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# ----------------------------------------------------------------
# 检索配置
# ----------------------------------------------------------------
#: 默认检索文档数量
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))

#: 检索相似度阈值：Top-1 余弦相似度低于该值时判定知识库资料不足，
#: 直接返回兜底提示而不让生成模型自由发挥，以抑制幻觉
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))

#: 相对分差阈值：与 Top-1 分数相差超过该值的文档视为噪声，不进入生成上下文
SCORE_MARGIN: float = float(os.getenv("SCORE_MARGIN", "0.12"))

# ---------------------------------------------------------------------------
# 生成模型配置（任务二）
# ---------------------------------------------------------------------------
LLM_MODEL_PATH: str = os.getenv(
    "LLM_MODEL_PATH",
    "./models/Qwen3-0.6B",
)

#: 本地模型最大新生成 token 数
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

#: 生成温度（本地默认贪心解码，温度不生效）
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# 多轮对话短期记忆配置（任务二）
# ---------------------------------------------------------------------------
#: 每个会话保留的最近历史轮数（一问一答为一轮）
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "3"))

# ---------------------------------------------------------------------------
# 服务配置（任务三）
# ---------------------------------------------------------------------------
DEFAULT_HOST: str = os.getenv("DEFAULT_HOST", "127.0.0.1")
DEFAULT_PORT: int = int(os.getenv("DEFAULT_PORT", "8000"))


def ensure_dirs() -> None:
    """创建运行期需要的目录（storage / outputs）。"""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

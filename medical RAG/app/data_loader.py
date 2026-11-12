"""医疗问答数据加载、清洗与知识库文档构建。"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from app import config

# 需要清洗的全部文本字段
TEXT_COLUMNS: list[str] = [
    "disease", "symptom", "drug", "department",
    "description", "question", "answer",
]

# 构建检索文档时的字段（标准答案不参与嵌入，留作评估对照）
DOC_FIELDS: list[str] = ["disease", "department", "symptom", "drug",
                         "description", "question"]

# 正则：匹配控制字符（\x00-\x1f 不可见字符、全角空格 \u3000）
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u3000]")
# 正则：连续空白合并为单个普通空格
_MULTI_SPACE_RE = re.compile(r"[ \t　]+")


def detect_encoding(path: str | Any, try_encodings: tuple[str, ...] = (
        "utf-8-sig", "utf-8", "gb18030", "gbk")) -> str:
    """依次尝试常见编码，返回第一个能读取的编码（中文 Windows 默认 GBK）。"""
    for encoding in try_encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                f.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    return "gb18030"  # 兜底：对中文 Windows 文件兼容性最好


def clean_text(value: Any) -> str:
    """清洗单个文本：NaN→空串 → 去控制字符 → 合并空白 → 去首尾空格。"""
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = _CONTROL_CHAR_RE.sub(" ", text)   # 不可见字符替换为空格
    text = _MULTI_SPACE_RE.sub(" ", text)    # 多个空格合并
    return text.strip()                      # 去首尾空格


def load_raw_data(path: str | Any | None = None) -> pd.DataFrame:
    """加载原始 CSV（自动探测编码）。"""
    csv_path = str(path or config.DATA_PATH)
    encoding = detect_encoding(csv_path)
    return pd.read_csv(csv_path, encoding=encoding)


def load_and_clean(path: str | Any | None = None) -> pd.DataFrame:
    """加载 + 清洗：逐字段清理 → 删空值 → 按(disease,question)去重。"""
    dataframe = load_raw_data(path)

    # 逐字段文本清理
    for column in TEXT_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(clean_text)

    # 删除 disease/description/question/answer 任一为空的记录
    before = len(dataframe)
    dataframe = dataframe[
        dataframe["disease"].ne("")
        & dataframe["description"].ne("")
        & dataframe["question"].ne("")
        & dataframe["answer"].ne("")
    ].reset_index(drop=True)
    removed_empty = before - len(dataframe)

    # 同一疾病的同一问题只保留一条
    before = len(dataframe)
    dataframe = dataframe.drop_duplicates(
        subset=["disease", "question"], keep="first"
    ).reset_index(drop=True)
    removed_dup = before - len(dataframe)

    # 把删除条数挂到 attrs 上，供统计函数使用
    dataframe.attrs["removed_empty"] = removed_empty
    dataframe.attrs["removed_dup"] = removed_dup
    return dataframe


def basic_stats(dataframe: pd.DataFrame) -> dict[str, Any]:
    """统计：记录总数、字段非空率、疾病/科室分布、清洗删除数。"""
    completeness = {
        column: round(float(dataframe[column].astype(str).str.len().gt(0).mean()), 4)
        for column in TEXT_COLUMNS if column in dataframe.columns
    }
    return {
        "记录总数": int(len(dataframe)),
        "字段数量": int(len(dataframe.columns)),
        "字段完整性(非空率)": completeness,
        "疾病种类数": int(dataframe["disease"].nunique()),
        "疾病样本分布": dataframe["disease"].value_counts().to_dict(),
        "推荐科室分布": dataframe["department"].value_counts().to_dict(),
        "清洗删除空值记录数": int(dataframe.attrs.get("removed_empty", 0)),
        "清洗删除重复记录数": int(dataframe.attrs.get("removed_dup", 0)),
    }


def _format_doc_content(row: pd.Series) -> str:
    """把一行数据拼成结构化文本，用于嵌入和作为生成上下文。"""
    field_labels = [
        ("disease", "疾病名称"),
        ("department", "推荐就诊科室"),
        ("symptom", "典型症状"),
        ("drug", "常用治疗药物"),
        ("description", "疾病描述"),
        ("question", "常见问题"),
    ]
    lines = [f"{label}：{row[field]}"
             for field, label in field_labels
             if field in row.index and str(row[field]).strip()]
    return "\n".join(lines)


def build_documents(dataframe: pd.DataFrame) -> list[Document]:
    """DataFrame → LangChain Document 列表。

    page_content 是拼接后的结构化文本，metadata 保留全部原始字段（含标准答案）。
    doc_id 行号与 FAISS 向量行号一一对应，检索时直接用来取文档。
    """
    documents: list[Document] = []
    for idx, row in dataframe.iterrows():
        metadata = {column: str(row[column]) for column in TEXT_COLUMNS
                    if column in dataframe.columns}
        metadata["doc_id"] = int(idx)
        documents.append(
            Document(page_content=_format_doc_content(row), metadata=metadata)
        )
    return documents

"""RAG 业务编排：检索 → 过滤 → Prompt → 生成 → 记忆。"""
from __future__ import annotations

import threading

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app import config
from app.llm import LocalLLM
from app.memory import ConversationMemory
from app.retriever import MedicalRetriever, RetrievedDoc, format_context

# 系统提示：约束模型只依据资料回答，不编造
SYSTEM_PROMPT = (
    "你是医疗健康问答助手。规则：\n"
    "1. 只能依据【参考资料】回答，不得编造资料中没有的信息；\n"
    "2. 回答准确、完整、条理清晰，必要时分点；\n"
    "3. 可以结合对话历史理解指代，但事实只能来自参考资料；\n"
    "4. 参考资料不足时说\"根据现有资料无法确定\"，并建议就医；\n"
    "5. 信息仅供科普，不能替代医生诊断。"
)

# 用户消息模板：把检索上下文与用户问题拼在一起
HUMAN_PROMPT = """【参考资料】
{context}

【用户问题】
{question}"""

# LangChain 模板：system 提示 + 多轮历史 + 当前(资料+问题)
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),           # 多轮对话历史占位
    ("human", HUMAN_PROMPT),
])


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """LangChain 消息对象 → {role, content} 列表，适配 LocalLLM.generate。"""
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role_map.get(m.type, m.type), "content": m.content}
            for m in messages]


class QAService:
    """RAG 问答服务单例。启动时加载检索器，生成模型首次问答才加载。"""

    _instance: "QAService | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> "QAService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        # 检索器：从 storage/ 加载，没有就自动清洗 + 建库
        self.retriever = MedicalRetriever.from_storage()
        self.memory = ConversationMemory()
        self._llm: LocalLLM | None = None
        self._initialized = True

    @property
    def llm(self) -> LocalLLM:
        """懒加载：第一次问答时才初始化生成模型（慢）。"""
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    self._llm = LocalLLM()
        return self._llm

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = config.DEFAULT_TOP_K):
        """仅检索，供 /qa/retrieve 接口使用。"""
        return self.retriever.retrieve(question, top_k=top_k)

    def answer(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = config.DEFAULT_TOP_K,
        use_history: bool = True,
        max_new_tokens: int | None = None,
    ) -> dict:
        """完整 RAG 问答：检索 → 过滤 → 生成 → 写入记忆。"""
        question = question.strip()

        # 1. 取该会话的历史（多轮指代扩展用）
        history = self.memory.get_history(session_id) if use_history else []

        # 2. 检索 Top-K
        #    多轮追问常含指代（"那应该用什么药"），把最近一轮用户问题拼进去
        #    扩展后的检索查询只用于检索，不进入生成 Prompt
        retrieval_query = question
        if history:
            last_user_q = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None,
            )
            if last_user_q:
                retrieval_query = f"{last_user_q} {question}"
        retrieved = self.retrieve(retrieval_query, top_k=top_k)

        # 3. 双重相似度过滤
        #    - 绝对阈值 0.55：知识库外问题直接拒答
        #    - 相对分差 0.12：剔除 Top-K 中的弱相关噪声（比如 Top-2 分数太低）
        references_payload = [
            {
                "title": doc.title, "content": doc.content,
                "score": round(doc.score, 4),
                "department": doc.metadata.get("department", ""),
            }
            for doc in retrieved
        ]
        top_score = retrieved[0].score if retrieved else 0.0
        cutoff = max(config.SIMILARITY_THRESHOLD, top_score - config.SCORE_MARGIN)
        effective_docs = [d for d in retrieved if d.score >= cutoff]

        # 知识库资料不足 → 兜底拒答，抑制小模型幻觉
        if not effective_docs:
            answer_text = (
                f"抱歉，知识库中未检索到相关资料（最高相似度 {top_score:.2f}）。"
                "建议前往正规医疗机构面诊，或补充更具体的描述再提问。"
            )
            if use_history:
                self.memory.add_turn(session_id, question, answer_text)
            return {
                "question": question, "answer": answer_text,
                "sources": [], "references": references_payload,
                "session_id": session_id,
            }

        # 4. 拼接有效文档为上下文
        context = format_context(effective_docs)

        # 5. LangChain 模板拼消息：system + history + (context + question)
        prompt_messages = QA_PROMPT.format_messages(
            context=context, question=question, history=history,
        )

        # 6. 调用生成模型
        answer_text = self.llm.generate(
            _messages_to_dicts(prompt_messages),
            max_new_tokens=max_new_tokens,
        )

        # 7. 写入短期记忆，供下一轮追问使用
        if use_history:
            self.memory.add_turn(session_id, question, answer_text)

        # 8. 参考来源：有效文档的 title，最多 3 个
        sources: list[str] = []
        for doc in effective_docs[:3]:
            if doc.title not in sources:
                sources.append(doc.title)

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "references": references_payload,
            "session_id": session_id,
        }

    def clear_history(self, session_id: str) -> None:
        """清空会话历史。"""
        self.memory.clear(session_id)

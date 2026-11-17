"""多轮对话短期记忆：按 session_id 隔离，定长队列自动淘汰旧消息。"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque

from app import config


class ConversationMemory:
    """会话级短期对话记忆。

    每个 session 存一个 deque(maxlen=2*max_turns)，
    一问一答各占一个槽位，超出窗口自动丢弃最旧的。
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self.max_turns = max_turns or config.MAX_HISTORY_TURNS
        self._store: dict[str, Deque[dict[str, str]]] = {}
        self._lock = threading.Lock()  # 保证 FastAPI 并发安全

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取历史消息列表（按时间升序）。"""
        with self._lock:
            return list(self._store.get(session_id, deque()))

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮问答；deque 自动淘汰旧的，不用手动判断长度。"""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=2 * self.max_turns)
            history = self._store[session_id]
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

    def clear(self, session_id: str) -> None:
        """清空指定会话。"""
        with self._lock:
            self._store.pop(session_id, None)

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID, uuid4

from app.services.exceptions import ChatRunNotFoundError


@dataclass(slots=True)
class ChatRunHandle:
    """表示一次流式聊天运行的可停止句柄。"""
    run_id: UUID
    conversation_id: UUID
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False


class ChatRunRegistry:
    def __init__(self) -> None:
        self._runs: dict[UUID, ChatRunHandle] = {}
        self._lock = RLock()

    def register_run(self, conversation_id: UUID) -> ChatRunHandle:
        """登记运行句柄，供后续停止请求按 run_id 查找。"""
        handle = ChatRunHandle(run_id=uuid4(), conversation_id=conversation_id)
        with self._lock:
            self._runs[handle.run_id] = handle
        return handle

    def request_stop(self, run_id: UUID) -> ChatRunHandle:
        """仅标记 stop_event，真正收尾由流式循环完成。"""
        with self._lock:
            handle = self._runs.get(run_id)
            if handle is None or handle.finished:
                raise ChatRunNotFoundError("chat run not found")
            handle.stop_event.set()
            return handle

    def finish_run(self, run_id: UUID) -> None:
        """运行结束后移除句柄，避免停止接口命中已完成任务。"""
        with self._lock:
            handle = self._runs.pop(run_id, None)
            if handle is not None:
                handle.finished = True

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


_registry = ChatRunRegistry()


def register_chat_run(conversation_id: UUID) -> ChatRunHandle:
    """对外暴露的注册入口，便于测试替换全局注册表。"""
    return _registry.register_run(conversation_id)


async def request_stop_chat_run(run_id: UUID | str) -> dict[str, str]:
    """兼容 UUID 和字符串输入，返回稳定的 API 响应载荷。"""
    normalized = run_id if isinstance(run_id, UUID) else UUID(str(run_id))
    handle = _registry.request_stop(normalized)
    return {"run_id": str(handle.run_id), "status": "stop_requested"}


def finish_chat_run(run_id: UUID) -> None:
    """对外暴露的结束入口。"""
    _registry.finish_run(run_id)


def clear_chat_runs() -> None:
    _registry.clear()

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID, uuid4

from app.services.exceptions import ChatRunNotFoundError


@dataclass(slots=True)
class ChatRunHandle:
    """表示一次流式聊天运行的可停止句柄。

    用于在流式处理、停止请求和后台收尾之间传递统一状态。
    """
    run_id: UUID
    conversation_id: UUID
    user_id: str
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False


class ChatRunRegistry:
    def __init__(self) -> None:
        """初始化运行注册表。

        Args:
            self: 注册表实例本身。
        """
        self._runs: dict[UUID, ChatRunHandle] = {}
        self._lock = RLock()

    def register_run(self, conversation_id: UUID, user_id: str = "") -> ChatRunHandle:
        """登记一个新的聊天运行句柄。

        生成 `run_id` 并把句柄放入注册表，供后续停止请求查找。

        Args:
            conversation_id: 会话 ID。
            user_id: 当前用户 ID。

        Returns:
            新创建的运行句柄。
        """
        handle = ChatRunHandle(run_id=uuid4(), conversation_id=conversation_id, user_id=user_id)
        with self._lock:
            self._runs[handle.run_id] = handle
        return handle

    def request_stop(self, run_id: UUID, user_id: str = "") -> ChatRunHandle:
        """请求停止指定运行。

        只设置 `stop_event`，由流式循环负责安全收尾并清理状态。

        Args:
            run_id: 运行 ID。
            user_id: 当前用户 ID。

        Returns:
            被请求停止的运行句柄。
        """
        with self._lock:
            handle = self._runs.get(run_id)
            if handle is None or handle.finished or handle.user_id != user_id:
                raise ChatRunNotFoundError("chat run not found")
            handle.stop_event.set()
            return handle

    def finish_run(self, run_id: UUID) -> None:
        """标记运行结束并移除句柄。

        避免停止接口在任务结束后继续命中已完成的运行。
        """
        with self._lock:
            handle = self._runs.pop(run_id, None)
            if handle is not None:
                handle.finished = True

    def clear(self) -> None:
        """清空所有运行句柄。

        用于测试隔离或进程级重置。
        """
        with self._lock:
            self._runs.clear()


_registry = ChatRunRegistry()


def register_chat_run(conversation_id: UUID, user_id: str = "") -> ChatRunHandle:
    """对外暴露运行注册入口。

    便于测试替换全局注册表实现。

    Args:
        conversation_id: 会话 ID。
        user_id: 当前用户 ID。

    Returns:
        新创建的运行句柄。
    """
    return _registry.register_run(conversation_id, user_id)


async def request_stop_chat_run(run_id: UUID | str, user_id: str = "") -> dict[str, str]:
    """请求停止聊天运行并返回标准响应。

    兼容 `UUID` 和字符串输入，统一输出可直接返回给 API 的载荷。

    Args:
        run_id: 运行 ID。
        user_id: 当前用户 ID。

    Returns:
        可直接返回给 API 的标准响应体。
    """
    normalized = run_id if isinstance(run_id, UUID) else UUID(str(run_id))
    handle = _registry.request_stop(normalized, user_id)
    return {"run_id": str(handle.run_id), "status": "stop_requested"}


def finish_chat_run(run_id: UUID) -> None:
    """对外暴露运行结束入口。

    将结束动作委托给全局注册表统一处理。
    """
    _registry.finish_run(run_id)


def clear_chat_runs() -> None:
    """清空全局聊天运行注册表。

    用于测试清理或重置进程状态。
    """
    _registry.clear()

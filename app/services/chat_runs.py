from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID, uuid4

from app.services.exceptions import ChatRunNotFoundError


@dataclass(slots=True)
class ChatRunHandle:
    run_id: UUID
    conversation_id: UUID
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False


class ChatRunRegistry:
    def __init__(self) -> None:
        self._runs: dict[UUID, ChatRunHandle] = {}
        self._lock = RLock()

    def register_run(self, conversation_id: UUID) -> ChatRunHandle:
        handle = ChatRunHandle(run_id=uuid4(), conversation_id=conversation_id)
        with self._lock:
            self._runs[handle.run_id] = handle
        return handle

    def request_stop(self, run_id: UUID) -> ChatRunHandle:
        with self._lock:
            handle = self._runs.get(run_id)
            if handle is None or handle.finished:
                raise ChatRunNotFoundError("chat run not found")
            handle.stop_event.set()
            return handle

    def finish_run(self, run_id: UUID) -> None:
        with self._lock:
            handle = self._runs.pop(run_id, None)
            if handle is not None:
                handle.finished = True

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


_registry = ChatRunRegistry()


def register_chat_run(conversation_id: UUID) -> ChatRunHandle:
    return _registry.register_run(conversation_id)


async def request_stop_chat_run(run_id: UUID | str) -> dict[str, str]:
    normalized = run_id if isinstance(run_id, UUID) else UUID(str(run_id))
    handle = _registry.request_stop(normalized)
    return {"run_id": str(handle.run_id), "status": "stop_requested"}


def finish_chat_run(run_id: UUID) -> None:
    _registry.finish_run(run_id)


def clear_chat_runs() -> None:
    _registry.clear()

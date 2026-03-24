from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, patch

from app.core.config import ConfigurationError, ModelEndpointSettings
from app.llm.base import ThinkingNotSupportedError, UpstreamServiceError
from app.schemas.chat import ChatRequest
from app.services.chat_runs import clear_chat_runs, request_stop_chat_run
from app.services.chat_service import stream_chat_events


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def wait_for_condition(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(0)


def fake_settings(*, provider: str = "anthropic") -> SimpleNamespace:
    return SimpleNamespace(
        memory_window_size=8,
        chat_endpoint=ModelEndpointSettings(
            provider=provider,
            api_key="test-key",
            base_url=None,
            model="test-model",
        ),
    )


@dataclass
class FakeConversation:
    id: UUID
    title: str
    summary: str | None = None
    summary_message_count: int = 0
    system_prompt: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class FakeMessage:
    id: int
    conversation_id: UUID
    role: str
    content: str
    sequence: int
    trace_steps: list[dict[str, object]] | None = None
    created_at: datetime = field(default_factory=utcnow)


class FakeRepository:
    def __init__(self):
        self.conversation: FakeConversation | None = None
        self.messages: list[FakeMessage] = []

    async def create_conversation(
        self,
        title: str,
        user_id: str | None = None,
        system_prompt: str | None = None,
    ) -> FakeConversation:
        self.conversation = FakeConversation(
            id=uuid4(),
            title=title,
            system_prompt=system_prompt,
        )
        return self.conversation

    async def get_conversation(self, conversation_id: UUID, *, include_deleted: bool = False):
        if self.conversation and self.conversation.id == conversation_id:
            return self.conversation
        return None

    async def add_message(
        self,
        conversation: FakeConversation,
        *,
        role: str,
        content: str,
        trace_steps: list[dict[str, object]] | None = None,
    ) -> FakeMessage:
        message = FakeMessage(
            id=len(self.messages) + 1,
            conversation_id=conversation.id,
            role=role,
            content=content,
            sequence=len(self.messages) + 1,
            trace_steps=trace_steps,
        )
        self.messages.append(message)
        conversation.updated_at = utcnow()
        return message

    async def get_message(self, message_id: int) -> FakeMessage | None:
        for message in self.messages:
            if message.id == message_id:
                return message
        return None

    async def update_message_trace(
        self,
        message: FakeMessage,
        *,
        trace_steps: list[dict[str, object]] | None,
    ) -> FakeMessage:
        message.trace_steps = trace_steps
        return message

    async def list_recent_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[FakeMessage]:
        messages = [item for item in self.messages if item.conversation_id == conversation_id]
        if before_sequence is not None:
            messages = [item for item in messages if item.sequence < before_sequence]
        return messages[-limit:]

    async def count_messages(self, conversation_id: UUID) -> int:
        return len([item for item in self.messages if item.conversation_id == conversation_id])

    async def update_title(self, conversation: FakeConversation, title: str) -> FakeConversation:
        conversation.title = title
        conversation.updated_at = utcnow()
        return conversation


class FakeSession:
    async def commit(self) -> None:
        return None


class FakeSessionFactory:
    def __init__(self):
        self.session = FakeSession()

    def __call__(self):
        session = self.session

        class _ContextManager:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _ContextManager()


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        clear_chat_runs()

    async def asyncTearDown(self):
        clear_chat_runs()

    async def test_thinking_disabled_returns_chunks_and_ignores_non_text_blocks(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "search", "step_id": "ignored-search", "query": "不应透出"}
                yield "你好！"
                yield " 我在。"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events = [event async for event in stream]

        self.assertEqual([name for name, _ in events], ["conversation", "chunk", "chunk", "done"])
        self.assertEqual(repository.messages[1].content, "你好！ 我在。")
        self.assertIsNone(repository.messages[1].trace_steps)

    async def test_first_round_generates_conversation_title_event_and_persists_title(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "text", "text": "RAG 是一种", "index": 0}
                yield {"type": "text", "text": "检索增强生成方法。", "index": 1}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch(
                "app.services.chat_service.generate_conversation_title",
                new=AsyncMock(return_value="RAG 入门"),
                create=True,
            ),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="帮我解释 RAG", thinking_enabled=False))
            events = [event async for event in stream]

        self.assertEqual(
            [name for name, _ in events],
            ["conversation", "chunk", "chunk", "conversation_title", "done"],
        )
        self.assertEqual(events[-2][1], {"conversation_id": str(repository.conversation.id), "title": "RAG 入门"})
        self.assertEqual(repository.conversation.title, "RAG 入门")
        self.assertEqual(repository.messages[1].content, "RAG 是一种检索增强生成方法。")

    async def test_existing_conversation_first_message_also_generates_title(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        conversation = await repository.create_conversation(title="新对话")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "text", "text": "这是首轮回答。", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", new=AsyncMock(return_value="首轮标题")),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(
                    conversation_id=conversation.id,
                    message="这是第一条消息",
                    thinking_enabled=False,
                )
            )
            events = [event async for event in stream]

        self.assertEqual(events[-2], ("conversation_title", {"conversation_id": str(conversation.id), "title": "首轮标题"}))
        self.assertEqual(repository.conversation.title, "首轮标题")

    async def test_non_first_round_does_not_generate_title(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        conversation = await repository.create_conversation(title="已有标题")
        await repository.add_message(conversation, role="user", content="上一轮问题")
        await repository.add_message(conversation, role="assistant", content="上一轮回答")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "text", "text": "这是后续回答。", "index": 0}

            return iterator()

        title_generator = AsyncMock(return_value="不应生成")

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", new=title_generator),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(
                    conversation_id=conversation.id,
                    message="继续说说实现细节",
                    thinking_enabled=False,
                )
            )
            events = [event async for event in stream]

        self.assertEqual([name for name, _ in events], ["conversation", "chunk", "done"])
        title_generator.assert_not_awaited()
        self.assertEqual(repository.conversation.title, "已有标题")

    async def test_title_generation_failure_degrades_without_interrupting_chat(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "text", "text": "正常回答。", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch(
                "app.services.chat_service.generate_conversation_title",
                new=AsyncMock(side_effect=ConfigurationError("TITLE_AGENT_API_KEY is not configured")),
            ),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="帮我解释 RAG", thinking_enabled=False))
            events = [event async for event in stream]

        self.assertEqual([name for name, _ in events], ["conversation", "chunk", "done"])
        self.assertEqual(events[-1][1]["status"], "completed")
        self.assertEqual(repository.conversation.title, "新对话")

    async def test_title_generation_upstream_failure_degrades_without_interrupting_chat(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {"type": "text", "text": "正常回答。", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch(
                "app.services.chat_service.generate_conversation_title",
                new=AsyncMock(side_effect=UpstreamServiceError("boom")),
            ),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="帮我解释 RAG", thinking_enabled=False))
            events = [event async for event in stream]

        self.assertEqual([name for name, _ in events], ["conversation", "chunk", "done"])
        self.assertEqual(repository.conversation.title, "新对话")

    async def test_empty_assistant_content_does_not_generate_title(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                if False:
                    yield None

            return iterator()

        title_generator = AsyncMock(return_value="不应生成")

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", new=title_generator),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="帮我解释 RAG", thinking_enabled=False))
            events = [event async for event in stream]

        self.assertEqual([name for name, _ in events], ["conversation", "done"])
        title_generator.assert_not_awaited()
        self.assertEqual(repository.conversation.title, "新对话")

    async def test_thinking_enabled_converts_thinking_and_search_to_trace_steps(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)
            self.assertEqual([(message.role, message.content) for message in messages], [("user", "你好")])

            async def iterator():
                yield {"type": "thinking", "thinking": "先分析用户意图。", "signature": "sig-1", "index": 0}
                yield {
                    "type": "search",
                    "step_id": "search-1",
                    "status": "success",
                    "title": "搜索资料",
                    "message": "先搜一下相关资料。",
                    "query": "你好",
                    "order": 2,
                    "kind": "result_list",
                }
                yield {"type": "text", "text": "你好", "index": 0}
                yield {"type": "text", "text": "！", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))
            events = [event async for event in stream]

        event_names = [name for name, _ in events]
        trace_events = [payload for name, payload in events if name == "trace_step"]
        self.assertEqual(
            event_names,
            ["conversation", "trace_step", "trace_step", "trace_step", "chunk", "chunk", "trace_done", "done"],
        )
        self.assertEqual(trace_events[0]["type"], "thinking")
        self.assertEqual(trace_events[0]["thinking"], "先分析用户意图。")
        self.assertEqual(trace_events[0]["status"], "running")
        self.assertEqual(trace_events[1]["type"], "thinking")
        self.assertEqual(trace_events[1]["status"], "success")
        self.assertEqual(trace_events[1]["thinking"], "先分析用户意图。")
        self.assertEqual(trace_events[2]["type"], "search")
        self.assertEqual(repository.messages[-1].content, "你好！")
        self.assertEqual(
            [(step["type"], step["status"]) for step in repository.messages[-1].trace_steps],
            [("thinking", "success"), ("search", "success")],
        )

    async def test_thinking_enabled_emits_thinking_trace_before_first_chunk_even_if_search_arrives_later(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield {"type": "thinking", "thinking": "先分析用户意图。", "signature": "sig-1", "index": 0}
                yield {"type": "text", "text": "你好", "index": 0}
                yield {
                    "type": "search",
                    "step_id": "search-1",
                    "status": "success",
                    "title": "搜索资料",
                    "message": "先搜一下相关资料。",
                    "query": "你好",
                    "order": 2,
                    "kind": "result_list",
                }

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))
            events = [event async for event in stream]

        self.assertEqual(
            [name for name, _ in events],
            ["conversation", "trace_step", "trace_step", "chunk", "trace_step", "trace_done", "done"],
        )
        self.assertEqual(events[1][1]["type"], "thinking")
        self.assertEqual(events[1][1]["status"], "running")
        self.assertEqual(events[2][1]["type"], "thinking")
        self.assertEqual(events[2][1]["status"], "success")
        self.assertEqual(events[3], ("chunk", {"content": "你好"}))
        self.assertEqual(events[4][1]["type"], "search")
        self.assertEqual(
            [(step["type"], step["status"]) for step in repository.messages[-1].trace_steps],
            [("thinking", "success"), ("search", "success")],
        )

    async def test_thinking_enabled_finishes_thinking_before_trace_done_when_stream_ends(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield {"type": "thinking", "thinking": "先分析用户意图。", "signature": "sig-1", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))
            events = [event async for event in stream]

        trace_events = [payload for name, payload in events if name == "trace_step"]
        self.assertEqual(
            [name for name, _ in events],
            ["conversation", "trace_step", "trace_step", "trace_done", "done"],
        )
        self.assertEqual(
            [(payload["type"], payload["status"]) for payload in trace_events],
            [("thinking", "running"), ("thinking", "success")],
        )
        self.assertEqual(len(repository.messages), 1)

    async def test_thinking_enabled_stop_before_reply_keeps_only_user_message(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        release_stream = asyncio.Event()
        stream_closed = asyncio.Event()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                try:
                    yield {"type": "thinking", "thinking": "先分析用户意图。", "signature": "sig-1", "index": 0}
                    await release_stream.wait()
                    yield {"type": "text", "text": "不应出现", "index": 1}
                finally:
                    stream_closed.set()

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))
            events: list[tuple[str, dict[str, object]]] = []

            async def consume():
                async for event in stream:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            await wait_for_condition(lambda: any(name == "trace_step" for name, _ in events))
            run_id = str(events[0][1]["run_id"])
            stop_response = await request_stop_chat_run(run_id)
            await asyncio.wait_for(consumer, timeout=1)
            await asyncio.wait_for(stream_closed.wait(), timeout=1)

        self.assertEqual(stop_response, {"run_id": run_id, "status": "stop_requested"})
        self.assertEqual(
            [payload["status"] for name, payload in events if name == "trace_step"],
            ["running"],
        )
        self.assertEqual(events[-2], ("trace_done", {"status": "stopped"}))
        self.assertEqual(events[-1], ("done", {"status": "stopped", "run_id": run_id}))
        self.assertEqual(len(repository.messages), 1)

    async def test_thinking_enabled_unsupported_provider_raises_before_stream_consumption(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings(provider="openai")),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch(
                "app.services.chat_service.validate_chat_capabilities",
                side_effect=ThinkingNotSupportedError("provider openai does not support thinking"),
            ),
        ):
            with self.assertRaises(ThinkingNotSupportedError) as context:
                await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))

        self.assertEqual(str(context.exception), "provider openai does not support thinking")
        self.assertEqual(len(repository.messages), 0)

    async def test_stop_request_persists_partial_answer_and_returns_stopped_done(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        stream_closed = asyncio.Event()
        release_stream = asyncio.Event()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            async def iterator():
                try:
                    yield {"type": "text", "text": "部分回答", "index": 0}
                    await release_stream.wait()
                finally:
                    stream_closed.set()

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events: list[tuple[str, dict[str, object]]] = []

            async def consume():
                async for event in stream:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            await wait_for_condition(lambda: len(events) >= 2)
            run_id = str(events[0][1]["run_id"])
            stop_response = await request_stop_chat_run(run_id)
            await asyncio.wait_for(consumer, timeout=1)
            await asyncio.wait_for(stream_closed.wait(), timeout=1)

        self.assertEqual(stop_response, {"run_id": run_id, "status": "stop_requested"})
        self.assertEqual(events[-1], ("done", {"status": "stopped", "run_id": run_id}))
        self.assertEqual(repository.messages[1].content, "部分回答")

    async def test_stop_request_before_first_chunk_keeps_only_user_message(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        stream_closed = asyncio.Event()
        release_stream = asyncio.Event()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            async def iterator():
                try:
                    await release_stream.wait()
                    yield {"type": "text", "text": "不应出现", "index": 0}
                finally:
                    stream_closed.set()

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=fake_settings()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events: list[tuple[str, dict[str, object]]] = []

            async def consume():
                async for event in stream:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0)
            run_id = str(events[0][1]["run_id"])
            await request_stop_chat_run(run_id)
            await asyncio.wait_for(consumer, timeout=1)
            await asyncio.wait_for(stream_closed.wait(), timeout=1)

        self.assertEqual(events, [("conversation", events[0][1]), ("done", {"status": "stopped", "run_id": run_id})])
        self.assertEqual(len(repository.messages), 1)


if __name__ == "__main__":
    unittest.main()

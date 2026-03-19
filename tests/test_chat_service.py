from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, patch

from app.llm.base import UpstreamServiceError
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
    content_blocks: list[dict[str, object]] | None = None
    reasoning_summary: str | None = None
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
        content_blocks: list[dict[str, object]] | None = None,
        reasoning_summary: str | None = None,
        trace_steps: list[dict[str, object]] | None = None,
    ) -> FakeMessage:
        message = FakeMessage(
            id=len(self.messages) + 1,
            conversation_id=conversation.id,
            role=role,
            content=content,
            sequence=len(self.messages) + 1,
            content_blocks=content_blocks,
            reasoning_summary=reasoning_summary,
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

    async def update_message_reasoning(
        self,
        message: FakeMessage,
        *,
        reasoning_summary: str | None,
        trace_steps: list[dict[str, object]] | None,
    ) -> FakeMessage:
        message.reasoning_summary = reasoning_summary
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

    async def test_thinking_disabled_simple_route_returns_reply_without_route_or_trace(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(return_value="简单问候")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield "你好！"
                yield " 我在。"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events = [event async for event in stream]
            await wait_for_condition(lambda: repository.conversation.title == "简单问候")

        event_names = [name for name, _ in events]
        self.assertEqual(event_names, ["conversation", "chunk", "chunk", "done"])
        self.assertIn("run_id", events[0][1])
        self.assertEqual(repository.messages[0].role, "user")
        self.assertEqual(repository.messages[1].role, "assistant")
        self.assertEqual(repository.messages[1].content, "你好！ 我在。")
        self.assertIsNone(repository.messages[1].trace_steps)
        self.assertIsNone(repository.messages[1].reasoning_summary)
        generate_title.assert_awaited_once()

    async def test_thinking_enabled_chat_path_generates_title_and_emits_trace_done(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(return_value="短期记忆规划")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield "先"
                yield "规划"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(message="帮我规划短期记忆", thinking_enabled=True)
            )
            events = [event async for event in stream]
            await wait_for_condition(lambda: repository.conversation.title == "短期记忆规划")

        event_names = [name for name, _ in events]
        run_id = events[0][1]["run_id"]
        self.assertEqual(events[0][0], "conversation")
        self.assertEqual(events[0][1]["title"], "新对话")
        self.assertIn("run_id", events[0][1])
        self.assertEqual(event_names.count("chunk"), 2)
        self.assertIn("trace_done", event_names)
        self.assertNotIn("reasoning_chunk", event_names)
        self.assertNotIn("reasoning_done", event_names)
        self.assertEqual(events[-1], ("done", {"status": "completed", "run_id": run_id}))
        self.assertEqual(repository.conversation.title, "短期记忆规划")
        self.assertEqual(repository.messages[-1].content, "先规划")
        self.assertIsNone(repository.messages[-1].reasoning_summary)
        self.assertIsNone(repository.messages[-1].trace_steps)
        generate_title.assert_awaited_once()

    async def test_thinking_enabled_emits_chain_trace_and_persists_structured_steps(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(return_value="问候标题")
        generate_thought_steps = AsyncMock(
            side_effect=[
                [
                    {
                        "title": "确定查询方向",
                        "message": "先搜索可用的 IP 信息来源。",
                    }
                ],
                [
                    {
                        "title": "确定查询方向",
                        "message": "先搜索可用的 IP 信息来源。",
                    },
                    {
                        "title": "准备整理结果",
                        "message": "准备汇总搜索和抓取结果后回答用户。",
                    },
                ],
            ]
        )

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield {
                    "type": "thinking",
                    "thinking": "先搜索可用的 IP 信息来源。",
                    "signature": "sig-1",
                    "index": 0,
                }
                yield {
                    "type": "thinking",
                    "thinking": " 再准备整理搜索和抓取结果。",
                    "signature": "sig-1",
                    "index": 0,
                }
                yield {
                    "type": "search",
                    "step_id": "search-1",
                    "parent_step_id": "assistant-thinking",
                    "status": "success",
                    "title": "搜索 IP 信息",
                    "message": "先搜索可用的 IP 查询站点。",
                    "query": "207.97.137.107 IP lookup",
                    "result_count": 2,
                    "order": 1,
                    "kind": "result_list",
                    "payload": {
                        "items": [
                            {
                                "title": "IP Address Lookup",
                                "url": "https://example.com/ip",
                                "domain": "example.com",
                                "snippet": "Lookup an IP",
                            },
                            {
                                "title": "ASN Lookup",
                                "url": "https://asn.example.com",
                                "domain": "asn.example.com",
                                "snippet": "Whois details",
                            },
                        ]
                    },
                }
                yield {
                    "type": "fetch",
                    "step_id": "fetch-1",
                    "parent_step_id": "search-1",
                    "status": "error",
                    "title": "抓取 IP 查询结果",
                    "message": "抓取 ipinfo 失败。",
                    "url": "https://ipinfo.io/207.97.137.107/json",
                    "order": 2,
                    "kind": "fetch_result",
                    "payload": {
                        "url": "https://ipinfo.io/207.97.137.107/json",
                        "status": "failed",
                        "http_status": 403,
                        "error_code": "HTTP403",
                        "error_message": "Forbidden",
                    },
                }
                yield {
                    "type": "retry",
                    "step_id": "retry-1",
                    "parent_step_id": "search-1",
                    "status": "running",
                    "title": "尝试另一种方式",
                    "message": "切换到另一个 IP 查询服务。",
                    "retry_of": "fetch-1",
                    "order": 3,
                    "kind": "retry",
                    "payload": {"reason": "primary_fetch_failed"},
                }
                yield {
                    "type": "text",
                    "text": "你好！",
                    "index": 1,
                }
                yield {
                    "type": "text",
                    "text": " 有什么我可以帮助你的吗？",
                    "index": 1,
                }

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_thought_steps", generate_thought_steps, create=True),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(message="你好", thinking_enabled=True)
            )
            events = [event async for event in stream]

        event_names = [name for name, _ in events]
        thought_events = [payload for name, payload in events if name == "thought_step"]
        trace_events = [payload for name, payload in events if name == "trace_step"]
        self.assertEqual(event_names.count("chunk"), 2)
        self.assertGreaterEqual(event_names.count("thought_step"), 3)
        self.assertGreaterEqual(event_names.count("trace_step"), 3)
        self.assertIn("trace_done", event_names)
        self.assertNotIn("reasoning_chunk", event_names)
        self.assertNotIn("reasoning_done", event_names)
        self.assertLess(event_names.index("thought_step"), event_names.index("chunk"))
        self.assertEqual(thought_events[0]["status"], "running")
        self.assertEqual(thought_events[1]["status"], "success")
        self.assertEqual(thought_events[2]["status"], "running")
        self.assertEqual(
            repository.messages[-1].content,
            "你好！ 有什么我可以帮助你的吗？",
        )
        self.assertEqual(
            repository.messages[-1].content_blocks,
            [
                {
                    "type": "text",
                    "text": "你好！ 有什么我可以帮助你的吗？",
                    "index": 1,
                },
            ],
        )
        self.assertIsNone(repository.messages[-1].reasoning_summary)
        self.assertEqual(repository.messages[-1].trace_steps[0]["type"], "thought")
        self.assertEqual(repository.messages[-1].trace_steps[0]["title"], "确定查询方向")
        self.assertEqual(repository.messages[-1].trace_steps[1]["title"], "准备整理结果")
        self.assertEqual(repository.messages[-1].trace_steps[2]["payload"]["items"][0]["domain"], "example.com")
        self.assertEqual(repository.messages[-1].trace_steps[3]["payload"]["http_status"], 403)
        self.assertEqual(repository.messages[-1].trace_steps[4]["retry_of"], "fetch-1")
        self.assertEqual(trace_events[0]["status"], "success")
        self.assertEqual(trace_events[-1]["status"], "running")
        self.assertEqual(generate_thought_steps.await_count, 2)

    async def test_thinking_enabled_falls_back_to_single_trace_step_when_thought_generation_fails(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield {
                    "type": "thinking",
                    "thinking": "先分析用户意图。",
                    "signature": "sig-1",
                    "index": 0,
                }
                yield {
                    "type": "text",
                    "text": "你好",
                    "index": 1,
                }

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch(
                "app.services.chat_service.generate_thought_steps",
                AsyncMock(side_effect=UpstreamServiceError("thought failed")),
                create=True,
            ),
            patch("app.services.chat_service.generate_conversation_title", new=AsyncMock(return_value="问候标题")),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=True))
            events = [event async for event in stream]

        event_names = [name for name, _ in events]
        fallback_trace = [payload for name, payload in events if name == "trace_step" and payload["type"] == "thought"]
        self.assertNotIn("thought_step", event_names)
        self.assertEqual(fallback_trace[0]["status"], "running")
        self.assertEqual(fallback_trace[-1]["status"], "success")
        self.assertLess(event_names.index("trace_step"), event_names.index("chunk"))

    async def test_existing_assistant_content_blocks_are_replayed_into_next_round(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        conversation = await repository.create_conversation(title="人工标题")
        await repository.add_message(conversation, role="user", content="你好")
        assistant_blocks = [
            {
                "type": "thinking",
                "thinking": "先打招呼。",
                "signature": "sig-1",
                "index": 0,
            },
            {
                "type": "text",
                "text": "你好！",
                "index": 1,
            },
        ]
        await repository.add_message(
            conversation,
            role="assistant",
            content="你好！",
            content_blocks=assistant_blocks,
        )
        generate_title = AsyncMock(return_value="不应覆盖")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)
            self.assertEqual(messages[-2].content_blocks, assistant_blocks)

            async def iterator():
                yield {"type": "text", "text": "继续说", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(conversation_id=conversation.id, message="继续", thinking_enabled=True)
            )
            events = [event async for event in stream]

        run_id = events[0][1]["run_id"]
        self.assertEqual(events[-1], ("done", {"status": "completed", "run_id": run_id}))
        self.assertEqual(repository.messages[-1].content, "继续说")
        self.assertEqual(
            repository.messages[-1].content_blocks,
            [{"type": "text", "text": "继续说", "index": 0}],
        )

    async def test_existing_non_default_title_is_not_overwritten(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        conversation = await repository.create_conversation(title="人工标题")
        await repository.add_message(conversation, role="user", content="旧问题")
        await repository.add_message(conversation, role="assistant", content="旧回答")
        generate_title = AsyncMock(return_value="不应覆盖")

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield "新回答"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(conversation_id=conversation.id, message="继续说说实现细节", thinking_enabled=True)
            )
            events = [event async for event in stream]

        event_names = [name for name, _ in events]
        self.assertNotIn("conversation_updated", event_names)
        self.assertIn("trace_done", event_names)
        self.assertEqual(repository.conversation.title, "人工标题")
        self.assertIsNone(repository.messages[-1].reasoning_summary)
        generate_title.assert_not_awaited()

    async def test_title_agent_failure_does_not_break_chat(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(side_effect=UpstreamServiceError("title upstream failed"))

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertTrue(thinking_enabled)

            async def iterator():
                yield "回答"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(
                ChatRequest(message="给这个会话起个名字", thinking_enabled=True)
            )
            events = [event async for event in stream]
            await wait_for_condition(lambda: generate_title.await_count == 1)

        event_names = [name for name, _ in events]
        run_id = events[0][1]["run_id"]
        self.assertEqual(events[-1], ("done", {"status": "completed", "run_id": run_id}))
        self.assertNotIn("conversation_updated", event_names)
        self.assertIn("trace_done", event_names)
        self.assertEqual(repository.conversation.title, "新对话")
        self.assertIsNone(repository.messages[-1].reasoning_summary)
        generate_title.assert_awaited_once()

    async def test_thinking_disabled_complex_route_returns_route_and_trace(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        planner_result = {
            "route": "complex",
            "plan": ["搜索相关资料", "抓取候选页面摘要"],
        }

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {
                    "type": "search",
                    "step_id": "search-1",
                    "status": "success",
                    "title": "搜索相关资料",
                    "message": "先搜一下。",
                    "query": "复杂任务",
                    "order": 1,
                    "kind": "result_list",
                    "payload": {"items": [{"title": "结果", "url": "https://example.com"}]},
                }
                yield {"type": "text", "text": "我先给你整理一下。", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value=planner_result)),
            patch("app.services.chat_service.generate_conversation_title", new=AsyncMock(return_value="复杂任务整理")),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="查一下这个 IP", thinking_enabled=False))
            events = [event async for event in stream]

        event_names = [name for name, _ in events]
        route_payload = next(payload for name, payload in events if name == "route")
        self.assertEqual(
            event_names,
            ["conversation", "route", "planner_done", "trace_step", "chunk", "trace_done", "done"],
        )
        self.assertIn("run_id", events[0][1])
        self.assertEqual(route_payload["route"], "complex")
        self.assertEqual(repository.messages[0].role, "user")
        self.assertEqual(repository.messages[1].role, "assistant")
        self.assertEqual(repository.messages[1].trace_steps[0]["type"], "search")
        self.assertEqual(repository.messages[1].content, "我先给你整理一下。")

    async def test_first_round_title_generation_runs_in_background_without_blocking_done(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        title_started = asyncio.Event()
        release_title = asyncio.Event()

        async def fake_generate_title(messages):
            title_started.set()
            await release_title.wait()
            return "简单问候"

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield "你好！"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", side_effect=fake_generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events: list[tuple[str, dict[str, object]]] = []

            async def consume():
                async for event in stream:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            await asyncio.wait_for(consumer, timeout=0.2)
            await asyncio.wait_for(title_started.wait(), timeout=1)

            self.assertEqual([name for name, _ in events], ["conversation", "chunk", "done"])
            self.assertEqual(repository.conversation.title, "新对话")

            release_title.set()
            await wait_for_condition(lambda: repository.conversation.title == "简单问候")

    async def test_background_title_generation_uses_first_round_snapshot_after_second_turn_starts(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        title_started = asyncio.Event()
        release_title = asyncio.Event()
        captured_messages: list[list[tuple[str, str]]] = []
        stream_call_count = 0

        async def fake_generate_title(messages):
            captured_messages.append([(message.role, message.content) for message in messages])
            title_started.set()
            await release_title.wait()
            return "首轮标题"

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            nonlocal stream_call_count
            stream_call_count += 1

            async def iterator():
                if stream_call_count == 1:
                    yield "首轮回答"
                else:
                    yield "第二轮回答"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", side_effect=fake_generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            first_stream = await stream_chat_events(ChatRequest(message="第一轮问题", thinking_enabled=False))
            first_events = [event async for event in first_stream]

            await asyncio.wait_for(title_started.wait(), timeout=1)

            second_stream = await stream_chat_events(
                ChatRequest(
                    conversation_id=repository.conversation.id,
                    message="第二轮问题",
                    thinking_enabled=False,
                )
            )
            second_events = [event async for event in second_stream]

            release_title.set()
            await wait_for_condition(lambda: repository.conversation.title == "首轮标题")

        self.assertEqual([name for name, _ in first_events], ["conversation", "chunk", "done"])
        self.assertEqual([name for name, _ in second_events], ["conversation", "chunk", "done"])
        self.assertEqual(
            captured_messages[0],
            [("user", "第一轮问题"), ("assistant", "首轮回答")],
        )

    async def test_background_title_generation_does_not_overwrite_manual_title_change(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        title_started = asyncio.Event()
        release_title = asyncio.Event()
        title_returned = asyncio.Event()

        async def fake_generate_title(messages):
            title_started.set()
            await release_title.wait()
            title_returned.set()
            return "不应覆盖"

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield "首轮回答"

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", side_effect=fake_generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="第一轮问题", thinking_enabled=False))
            events = [event async for event in stream]

            await asyncio.wait_for(title_started.wait(), timeout=1)
            repository.conversation.title = "人工标题"

            release_title.set()
            await asyncio.wait_for(title_returned.wait(), timeout=1)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        self.assertEqual([name for name, _ in events], ["conversation", "chunk", "done"])
        self.assertEqual(repository.conversation.title, "人工标题")

    async def test_thinking_disabled_agent_route_keeps_tools_and_returns_trace(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        planner_result = {
            "route": "agent",
            "plan": ["搜索相关资料", "抓取候选页面摘要"],
            "tools": ["web_search", "http_fetch"],
        }

        async def fake_build_chat_stream(messages, *, thinking_enabled=False):
            self.assertFalse(thinking_enabled)

            async def iterator():
                yield {
                    "type": "tool_result",
                    "step_id": "tool-1",
                    "status": "success",
                    "title": "工具结果",
                    "message": "网页搜索已完成。",
                    "tool_name": "web_search",
                    "order": 1,
                    "kind": "tool_card",
                    "payload": {"items": [{"title": "结果", "url": "https://example.com"}]},
                }
                yield {"type": "text", "text": "我先查到这些结果。", "index": 0}

            return iterator()

        with (
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value=planner_result)),
            patch("app.services.chat_service.generate_conversation_title", new=AsyncMock(return_value="Agent 查询")),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="查一下这个 IP", thinking_enabled=False))
            events = [event async for event in stream]

        route_payload = next(payload for name, payload in events if name == "route")
        self.assertEqual(route_payload["route"], "agent")
        self.assertEqual(route_payload["tools"], ["web_search", "http_fetch"])
        self.assertEqual(repository.messages[1].trace_steps[0]["tool_name"], "web_search")

    async def test_stop_request_persists_partial_answer_and_returns_stopped_done(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(return_value="不应生成")
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
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
            patch("app.services.chat_service.refresh_summary_if_needed", new=AsyncMock()),
        ):
            stream = await stream_chat_events(ChatRequest(message="你好", thinking_enabled=False))
            events: list[tuple[str, dict[str, object]]] = []

            async def consume():
                async for event in stream:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            while len(events) < 2:
                await asyncio.sleep(0)

            run_id = str(events[0][1]["run_id"])
            stop_response = await request_stop_chat_run(run_id)
            await asyncio.wait_for(consumer, timeout=1)
            await asyncio.wait_for(stream_closed.wait(), timeout=1)

        event_names = [name for name, _ in events]
        self.assertEqual(stop_response, {"run_id": run_id, "status": "stop_requested"})
        self.assertEqual(event_names, ["conversation", "chunk", "done"])
        self.assertEqual(events[-1], ("done", {"status": "stopped", "run_id": run_id}))
        self.assertEqual(repository.messages[0].role, "user")
        self.assertEqual(repository.messages[1].role, "assistant")
        self.assertEqual(repository.messages[1].content, "部分回答")
        generate_title.assert_not_awaited()

    async def test_stop_request_before_first_chunk_keeps_only_user_message(self):
        repository = FakeRepository()
        session_factory = FakeSessionFactory()
        generate_title = AsyncMock(return_value="不应生成")
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
            patch("app.services.chat_service.get_settings", return_value=type("S", (), {"memory_window_size": 8})()),
            patch("app.services.chat_service.get_session_factory", return_value=session_factory),
            patch("app.services.chat_service.ConversationRepository", side_effect=lambda session: repository),
            patch("app.services.chat_service.build_chat_stream", side_effect=fake_build_chat_stream),
            patch("app.services.chat_service.plan_execution_route", new=AsyncMock(return_value={"route": "simple"})),
            patch("app.services.chat_service.generate_conversation_title", generate_title),
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
        self.assertEqual(repository.messages[0].role, "user")
        generate_title.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

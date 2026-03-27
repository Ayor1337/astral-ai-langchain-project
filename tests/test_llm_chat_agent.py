import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.core.config import ModelEndpointSettings
from app.llm.agents.chat import build_chat_stream
from app.schemas.chat import ChatMessage


def fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        chat_endpoint=ModelEndpointSettings(
            provider="anthropic",
            api_key="test-key",
            base_url=None,
            model="test-model",
        )
    )


class FakeAgent:
    def __init__(self, events):
        self.events = events
        self.calls: list[tuple[dict[str, object], object]] = []

    async def astream(self, payload, stream_mode):
        self.calls.append((payload, stream_mode))
        for event in self.events:
            yield event


class ChatAgentStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_chat_stream_uses_messages_mode_for_plain_text_streaming(self):
        agent = FakeAgent(
            [
                (AIMessageChunk(content="Hel"), {"langgraph_node": "model"}),
                (AIMessageChunk(content="lo"), {"langgraph_node": "model"}),
            ]
        )

        with patch("app.llm.agents.chat.create_chat_agent", return_value=agent):
            stream = await build_chat_stream(
                [ChatMessage(role="user", content="hello")],
                endpoint=fake_settings().chat_endpoint,
                thinking_enabled=False,
            )
            blocks = [block async for block in stream]

        self.assertEqual(agent.calls[0][1], "messages")
        self.assertEqual(
            blocks,
            [
                {"type": "text", "text": "Hel", "index": 0},
                {"type": "text", "text": "lo", "index": 0},
            ],
        )

    async def test_build_chat_stream_uses_messages_for_text_and_updates_for_trace(self):
        agent = FakeAgent(
            [
                (
                    "messages",
                    (
                        AIMessageChunk(
                            content=[
                                {
                                    "type": "thinking",
                                    "thinking": "先分析用户问题。",
                                    "signature": "sig-1",
                                    "index": 0,
                                }
                            ]
                        ),
                        {"langgraph_node": "model"},
                    ),
                ),
                ("messages", (AIMessageChunk(content="Hel"), {"langgraph_node": "model"})),
                (
                    "updates",
                    {
                        "model": {
                            "messages": [
                                AIMessage(
                                    content=[
                                        {
                                            "type": "thinking",
                                            "thinking": "不应重复的 thinking",
                                            "signature": "sig-1",
                                            "index": 0,
                                        },
                                        {"type": "text", "text": "Hello", "index": 0},
                                    ]
                                )
                            ],
                        }
                    },
                ),
                (
                    "updates",
                    {
                        "tools": {
                            "messages": [ToolMessage(content="42", tool_call_id="call-1", name="search")],
                        }
                    },
                ),
                ("messages", (AIMessageChunk(content="lo"), {"langgraph_node": "model"})),
            ]
        )

        with patch("app.llm.agents.chat.create_chat_agent", return_value=agent):
            stream = await build_chat_stream(
                [ChatMessage(role="user", content="hello")],
                endpoint=fake_settings().chat_endpoint,
                thinking_enabled=True,
            )
            blocks = [block async for block in stream]

        self.assertEqual(agent.calls[0][1], ["messages", "updates"])
        self.assertEqual(
            blocks,
            [
                {
                    "type": "thinking",
                    "thinking": "先分析用户问题。",
                    "signature": "sig-1",
                    "index": 0,
                },
                {"type": "text", "text": "Hel", "index": 0},
                {
                    "type": "tool_result",
                    "step_id": "call-1",
                    "tool_name": "search",
                    "output_json": "42",
                },
                {"type": "text", "text": "lo", "index": 0},
            ],
        )

    async def test_build_chat_stream_uses_explicit_endpoint_instead_of_global_settings(self):
        endpoint = ModelEndpointSettings(
            provider="anthropic",
            api_key="explicit-key",
            base_url="https://example.com",
            model="explicit-model",
        )
        agent = FakeAgent([])

        with patch("app.llm.agents.chat.create_chat_agent", return_value=agent) as create_agent_mock:
            stream = await build_chat_stream(
                [ChatMessage(role="user", content="hello")],
                endpoint=endpoint,
                thinking_enabled=True,
            )
            blocks = [block async for block in stream]

        self.assertEqual(blocks, [])
        create_agent_mock.assert_called_once_with(
            endpoint=endpoint,
            thinking_enabled=True,
            search_enabled=False,
        )

    async def test_build_chat_stream_emits_search_blocks_when_search_enabled_without_thinking(self):
        agent = FakeAgent(
            [
                (
                    "updates",
                    {
                        "model": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "name": "web_search",
                                            "args": {"query": "Astral AI 最新消息"},
                                            "id": "search-1",
                                        }
                                    ],
                                )
                            ]
                        }
                    },
                ),
                (
                    "updates",
                    {
                        "tools": {
                            "messages": [
                                ToolMessage(
                                    content='{"query":"Astral AI 最新消息","results":[{"title":"Astral AI","url":"https://example.com/astral","snippet":"Latest update"}]}',
                                    tool_call_id="search-1",
                                    name="web_search",
                                )
                            ],
                        }
                    },
                ),
                (AIMessageChunk(content="这里是答案[1]"), {"langgraph_node": "model"}),
            ]
        )

        with patch("app.llm.agents.chat.create_chat_agent", return_value=agent):
            stream = await build_chat_stream(
                [ChatMessage(role="user", content="Astral AI 最新消息")],
                endpoint=fake_settings().chat_endpoint,
                thinking_enabled=False,
                search_enabled=True,
            )
            blocks = [block async for block in stream]

        self.assertEqual(
            blocks,
            [
                {
                    "type": "search",
                    "step_id": "search-1",
                    "query": "Astral AI 最新消息",
                    "status": "running",
                    "kind": "result_list",
                    "message": "正在联网搜索。",
                },
                {
                    "type": "search",
                    "step_id": "search-1",
                    "query": "Astral AI 最新消息",
                    "status": "success",
                    "kind": "result_list",
                    "result_count": 1,
                    "payload": {
                        "results": [
                            {
                                "title": "Astral AI",
                                "url": "https://example.com/astral",
                                "snippet": "Latest update",
                            }
                        ]
                    },
                },
                {"type": "text", "text": "这里是答案[1]", "index": 0},
            ],
        )

    async def test_build_chat_stream_does_not_treat_tool_message_as_text_chunk(self):
        agent = FakeAgent(
            [
                (
                    "messages",
                    (
                        ToolMessage(
                            content='{"query":"1000日元兑换人民币汇率","results":[{"title":"1000日元等于多少人民币 - IP.cn","url":"https://www.ip.cn:8443/huilv/JPY-CNY/1000.html","snippet":"1000日元=44.683人民币"}]}',
                            tool_call_id="search-1",
                            name="web_search",
                        ),
                        {"langgraph_node": "tools"},
                    ),
                ),
                ("messages", (AIMessageChunk(content="1000日元约合人民币 44.68 元。[1]"), {"langgraph_node": "model"})),
            ]
        )

        with patch("app.llm.agents.chat.create_chat_agent", return_value=agent):
            stream = await build_chat_stream(
                [ChatMessage(role="user", content="1000日元兑换人民币汇率")],
                endpoint=fake_settings().chat_endpoint,
                thinking_enabled=False,
                search_enabled=True,
            )
            blocks = [block async for block in stream]

        self.assertEqual(
            blocks,
            [
                {"type": "text", "text": "1000日元约合人民币 44.68 元。[1]", "index": 0},
            ],
        )

if __name__ == "__main__":
    unittest.main()

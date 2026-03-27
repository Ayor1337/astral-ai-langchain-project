from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.core.config import ModelEndpointSettings
from app.llm.agents.chat import build_chat_stream
from app.schemas.chat import ChatMessage


@pytest.mark.anyio
async def test_build_chat_stream_emits_text_and_thinking_blocks():
    class FakeAgent:
        async def astream(self, payload, *, stream_mode):
            assert stream_mode == ["messages", "updates"]
            yield (
                "messages",
                (
                    AIMessageChunk(
                        content=[{"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0}]
                    ),
                    {"langgraph_node": "model"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="你"),
                    {"langgraph_node": "model"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="好"),
                    {"langgraph_node": "model"},
                ),
            )
            yield (
                "updates",
                {
                    "model": {
                        "messages": [AIMessage(content=[{"type": "text", "text": "你好", "index": 1}])]
                    }
                },
            )

    with (
        patch("app.llm.agents.chat.create_chat_agent", return_value=FakeAgent()),
    ):
        stream = await build_chat_stream(
            [ChatMessage(role="user", content="你好")],
            endpoint=ModelEndpointSettings(
                provider="anthropic",
                api_key="chat-key",
                base_url=None,
                model="claude-chat-model",
            ),
            thinking_enabled=True,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == [
        {"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0},
        {"type": "text", "text": "你", "index": 0},
        {"type": "text", "text": "好", "index": 0},
    ]


@pytest.mark.anyio
async def test_build_chat_stream_passthroughs_custom_trace_blocks():
    class FakeAgent:
        async def astream(self, payload, *, stream_mode):
            assert stream_mode == ["messages", "updates"]
            yield (
                "updates",
                {
                    "model": {
                        "messages": [
                            AIMessage(content=[{"type": "tool_call", "step_id": "tool-1", "tool_name": "web_search"}])
                        ]
                    }
                },
            )

    with (
        patch("app.llm.agents.chat.create_chat_agent", return_value=FakeAgent()),
    ):
        stream = await build_chat_stream(
            [ChatMessage(role="user", content="查 IP")],
            endpoint=ModelEndpointSettings(
                provider="anthropic",
                api_key="chat-key",
                base_url=None,
                model="claude-chat-model",
            ),
            thinking_enabled=True,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == [{"type": "tool_call", "step_id": "tool-1", "tool_name": "web_search"}]


@pytest.mark.anyio
async def test_build_chat_stream_executes_add_tool_and_streams_final_text():
    class FakeAgent:
        async def astream(self, payload, *, stream_mode):
            assert stream_mode == ["messages", "updates"]
            yield (
                "updates",
                {
                    "model": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call-1"}],
                            )
                        ]
                    }
                },
            )
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(content='{"result": 5}', name="add", tool_call_id="call-1")
                        ]
                    }
                },
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="5"),
                    {"langgraph_node": "model"},
                ),
            )
            yield (
                "updates",
                {
                    "model": {
                        "messages": [AIMessage(content=[{"type": "text", "text": "5", "index": 0}])]
                    }
                },
            )

    with (
        patch("app.llm.agents.chat.create_chat_agent", return_value=FakeAgent()),
    ):
        stream = await build_chat_stream(
            [ChatMessage(role="user", content="2+3 等于几？")],
            endpoint=ModelEndpointSettings(
                provider="anthropic",
                api_key="chat-key",
                base_url=None,
                model="claude-chat-model",
            ),
            thinking_enabled=True,
        )
        chunks = [chunk async for chunk in stream]

    assert [chunk["type"] for chunk in chunks] == ["tool_call", "tool_result", "text"]
    assert chunks[0]["tool_name"] == "add"
    assert chunks[0]["input_json"] == '{"a":2,"b":3}'
    assert chunks[1]["tool_name"] == "add"
    assert chunks[1]["output_json"] == '{"result": 5}'
    assert chunks[2]["text"] == "5"

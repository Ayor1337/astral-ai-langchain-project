from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import ModelEndpointSettings
from app.llm.base import build_chat_stream
from app.schemas.chat import ChatMessage


@pytest.mark.anyio
async def test_build_chat_stream_emits_text_and_thinking_blocks():
    async def fake_astream(messages):
        yield SimpleNamespace(content=[{"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0}])
        yield SimpleNamespace(content=[{"type": "text", "text": "你好", "index": 1}])

    model = SimpleNamespace(astream=fake_astream)

    with (
        patch(
            "app.llm.base.get_settings",
            return_value=SimpleNamespace(
                chat_endpoint=ModelEndpointSettings(
                    provider="anthropic",
                    api_key="chat-key",
                    base_url=None,
                    model="claude-chat-model",
                )
            ),
        ),
        patch("app.llm.base.create_chat_model", return_value=model),
    ):
        stream = await build_chat_stream([ChatMessage(role="user", content="你好")], thinking_enabled=True)
        chunks = [chunk async for chunk in stream]

    assert chunks == [
        {"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0},
        {"type": "text", "text": "你好", "index": 1},
    ]


@pytest.mark.anyio
async def test_build_chat_stream_passthroughs_custom_trace_blocks():
    async def fake_astream(messages):
        yield SimpleNamespace(content=[{"type": "tool_call", "step_id": "tool-1", "tool_name": "web_search"}])

    model = SimpleNamespace(astream=fake_astream)

    with (
        patch(
            "app.llm.base.get_settings",
            return_value=SimpleNamespace(
                chat_endpoint=ModelEndpointSettings(
                    provider="anthropic",
                    api_key="chat-key",
                    base_url=None,
                    model="claude-chat-model",
                )
            ),
        ),
        patch("app.llm.base.create_chat_model", return_value=model),
    ):
        stream = await build_chat_stream([ChatMessage(role="user", content="查 IP")], thinking_enabled=True)
        chunks = [chunk async for chunk in stream]

    assert chunks == [{"type": "tool_call", "step_id": "tool-1", "tool_name": "web_search"}]

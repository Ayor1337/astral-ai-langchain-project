from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.core.config import ConfigurationError, ModelEndpointSettings
from app.llm.agents.titile import generate_conversation_title
from app.llm.exceptions import UpstreamServiceError


def fake_settings(*, title_agent_endpoint: ModelEndpointSettings | None):
    return SimpleNamespace(title_agent_endpoint=title_agent_endpoint)


def fake_endpoint() -> ModelEndpointSettings:
    return ModelEndpointSettings(
        provider="anthropic",
        api_key="title-key",
        base_url="https://anthropic.example.com",
        model="claude-title-model",
    )


@pytest.mark.anyio
async def test_generate_conversation_title_uses_only_user_message_and_extracts_text():
    agent = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content=[{"type": "text", "text": "搜索资料", "index": 0}])
                ]
            }
        )
    )

    with (
        patch("app.llm.agents.titile.get_settings", return_value=fake_settings(title_agent_endpoint=fake_endpoint())),
        patch("app.llm.agents.titile.create_title_agent", return_value=agent),
    ):
        title = await generate_conversation_title(
            user_message="帮我查一下 RAG 是什么",
        )

    assert title == "搜索资料"
    agent.ainvoke.assert_awaited_once()
    prompt = agent.ainvoke.await_args.args[0]["messages"][0].content
    assert "用户：帮我查一下 RAG 是什么" in prompt
    assert "助手：" not in prompt


@pytest.mark.anyio
async def test_generate_conversation_title_normalizes_model_output():
    agent = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content=[{"type": "text", "text": '标题："RAG 入门指南"\n补充说明', "index": 0}])
                ]
            }
        )
    )

    with (
        patch("app.llm.agents.titile.get_settings", return_value=fake_settings(title_agent_endpoint=fake_endpoint())),
        patch("app.llm.agents.titile.create_title_agent", return_value=agent),
    ):
        title = await generate_conversation_title(
            user_message="帮我查一下 RAG 是什么",
        )

    assert title == "RAG 入门指南"


@pytest.mark.anyio
async def test_generate_conversation_title_falls_back_when_model_returns_empty_text():
    agent = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content=[{"type": "text", "text": '""', "index": 0}])
                ]
            }
        )
    )

    with (
        patch("app.llm.agents.titile.get_settings", return_value=fake_settings(title_agent_endpoint=fake_endpoint())),
        patch("app.llm.agents.titile.create_title_agent", return_value=agent),
    ):
        title = await generate_conversation_title(
            user_message="帮我查一下 RAG 是什么",
        )

    assert title == "新对话"


@pytest.mark.anyio
async def test_generate_conversation_title_wraps_agent_errors():
    agent = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))

    with (
        patch("app.llm.agents.titile.get_settings", return_value=fake_settings(title_agent_endpoint=fake_endpoint())),
        patch("app.llm.agents.titile.create_title_agent", return_value=agent),
        pytest.raises(UpstreamServiceError, match="boom"),
    ):
        await generate_conversation_title(
            user_message="帮我查一下 RAG 是什么",
        )


@pytest.mark.anyio
async def test_generate_conversation_title_requires_title_agent_configuration():
    with (
        patch("app.llm.agents.titile.get_settings", return_value=fake_settings(title_agent_endpoint=None)),
        pytest.raises(ConfigurationError, match="TITLE_AGENT_API_KEY"),
    ):
        await generate_conversation_title(
            user_message="帮我查一下 RAG 是什么",
        )

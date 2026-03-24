from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.llm.agents.summary import generate_summary
from app.llm.exceptions import UpstreamServiceError
from app.schemas.chat import ChatMessage


@pytest.mark.anyio
async def test_generate_summary_returns_previous_summary_when_no_messages():
    summary = await generate_summary(previous_summary="已有摘要", messages=[])

    assert summary == "已有摘要"


@pytest.mark.anyio
async def test_generate_summary_uses_summary_agent_and_extracts_text():
    agent = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content=[{"type": "text", "text": "更新后的摘要", "index": 0}])
                ]
            }
        )
    )

    with patch("app.llm.agents.summary.create_summary_agent", return_value=agent):
        summary = await generate_summary(
            previous_summary="旧摘要",
            messages=[ChatMessage(role="user", content="帮我总结一下")],
        )

    assert summary == "更新后的摘要"
    agent.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_generate_summary_wraps_agent_errors():
    agent = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))

    with (
        patch("app.llm.agents.summary.create_summary_agent", return_value=agent),
        pytest.raises(UpstreamServiceError, match="boom"),
    ):
        await generate_summary(
            previous_summary=None,
            messages=[ChatMessage(role="user", content="帮我总结一下")],
        )

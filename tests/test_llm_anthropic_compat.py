from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.llm.base import build_chat_stream
from app.llm.planner_agent import plan_execution_route
from app.llm.reasoning_agent import generate_reasoning_summary, generate_thought_steps
from app.llm.title_agent import generate_conversation_title
from app.schemas.chat import ChatMessage


@pytest.mark.anyio
async def test_generate_conversation_title_extracts_text_block_only():
    response = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "先总结首轮问答。", "signature": "sig-1"},
            {"type": "text", "text": "问候交流"},
        ]
    )

    async def fake_ainvoke(prompt):
        return response

    model = SimpleNamespace(ainvoke=fake_ainvoke)

    with patch("app.llm.title_agent.ChatAnthropic", return_value=model):
        title = await generate_conversation_title(
            [
                ChatMessage(role="user", content="你好"),
                ChatMessage(role="assistant", content="你好！"),
            ]
        )

    assert title == "问候交流"


@pytest.mark.anyio
async def test_generate_reasoning_summary_extracts_text_block_only():
    response = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "先提炼高层说明。", "signature": "sig-1"},
            {"type": "text", "text": "助手识别问候意图并礼貌回应。"},
        ]
    )

    async def fake_ainvoke(prompt):
        return response

    model = SimpleNamespace(ainvoke=fake_ainvoke)

    with patch("app.llm.reasoning_agent.ChatAnthropic", return_value=model):
        summary = await generate_reasoning_summary(
            user_message="你好",
            assistant_message="你好！有什么我可以帮助你的吗？",
        )

    assert summary == "助手识别问候意图并礼貌回应。"


@pytest.mark.anyio
async def test_generate_thought_steps_parses_json_text_block():
    response = SimpleNamespace(
        content=[
            {
                "type": "text",
                "text": '[{"title":"确定查询方向","message":"先搜索可用的 IP 信息来源。"},{"title":"准备整理结果","message":"准备汇总搜索和抓取结果后回答用户。"}]',
            }
        ]
    )

    async def fake_ainvoke(prompt):
        return response

    model = SimpleNamespace(ainvoke=fake_ainvoke)

    with patch("app.llm.reasoning_agent.ChatAnthropic", return_value=model):
        steps = await generate_thought_steps(
            user_message="查一下 207.97.137.107",
            raw_thinking="先搜索可用的 IP 信息来源。再准备整理搜索和抓取结果。",
            existing_steps=[],
        )

    assert steps == [
        {"title": "确定查询方向", "message": "先搜索可用的 IP 信息来源。"},
        {"title": "准备整理结果", "message": "准备汇总搜索和抓取结果后回答用户。"},
    ]


@pytest.mark.anyio
async def test_build_chat_stream_emits_text_and_thinking_blocks():
    async def fake_astream(messages):
        yield SimpleNamespace(content=[{"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0}])
        yield SimpleNamespace(content=[{"type": "text", "text": "你好", "index": 1}])

    model = SimpleNamespace(astream=fake_astream)

    with patch("app.llm.base._create_model", return_value=model):
        stream = await build_chat_stream([ChatMessage(role="user", content="你好")], thinking_enabled=True)
        chunks = [chunk async for chunk in stream]

    assert chunks == [
        {"type": "thinking", "thinking": "先分析。", "signature": "sig-1", "index": 0},
        {"type": "text", "text": "你好", "index": 1},
    ]


@pytest.mark.anyio
async def test_build_chat_stream_passthroughs_custom_trace_blocks():
    async def fake_astream(messages):
        yield SimpleNamespace(content=[{"type": "search", "step_id": "search-1", "query": "ip lookup", "order": 1}])

    model = SimpleNamespace(astream=fake_astream)

    with patch("app.llm.base._create_model", return_value=model):
        stream = await build_chat_stream([ChatMessage(role="user", content="查 IP")], thinking_enabled=True)
        chunks = [chunk async for chunk in stream]

    assert chunks == [{"type": "search", "step_id": "search-1", "query": "ip lookup", "order": 1}]


@pytest.mark.anyio
async def test_plan_execution_route_normalizes_agent_payload():
    response = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "先判断需要工具。", "signature": "sig-1"},
            {
                "type": "text",
                "text": '```json\n{"route":"complex_with_tools","plan":["搜索资料","抓取详情"],"tools":["web_search","http_fetch"],"extra":"ignore"}\n```',
            },
        ]
    )

    async def fake_ainvoke(prompt):
        return response

    model = SimpleNamespace(ainvoke=fake_ainvoke)

    with patch("app.llm.planner_agent.ChatAnthropic", return_value=model):
        result = await plan_execution_route(message="查一下这个 IP")

    assert result == {
        "route": "agent",
        "plan": ["搜索资料", "抓取详情"],
        "tools": ["web_search", "http_fetch"],
    }


@pytest.mark.anyio
async def test_plan_execution_route_uses_minimax_m2_model():
    response = SimpleNamespace(content=[{"type": "text", "text": '{"route":"simple"}'}])

    async def fake_ainvoke(prompt):
        return response

    model = SimpleNamespace(ainvoke=fake_ainvoke)

    with patch("app.llm.planner_agent.ChatAnthropic", return_value=model) as mocked_chat_anthropic:
        await plan_execution_route(message="你好")

    assert mocked_chat_anthropic.call_args.kwargs["model"] == "MiniMax-M2"

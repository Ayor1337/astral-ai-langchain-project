from app.schemas.chat import ChatMessage
from app.services.memory_service import build_context_messages, should_refresh_summary


def test_build_context_messages_includes_summary_and_current_message():
    recent_messages = [
        ChatMessage(role="user", content="旧问题"),
        ChatMessage(role="assistant", content="旧回答"),
    ]

    result = build_context_messages(
        system_prompt="你是助手",
        summary="用户正在规划短期记忆功能。",
        recent_messages=recent_messages,
        current_message="继续实现",
    )

    assert result[0].role == "system"
    assert result[0].content == "你是助手"
    assert result[1].role == "system"
    assert "用户正在规划短期记忆功能" in result[1].content
    assert result[-1].role == "user"
    assert result[-1].content == "继续实现"


def test_should_refresh_summary_when_message_count_exceeds_trigger():
    assert should_refresh_summary(total_messages=13, trigger=12) is True
    assert should_refresh_summary(total_messages=12, trigger=12) is False

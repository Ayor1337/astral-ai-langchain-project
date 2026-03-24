from collections.abc import Sequence

from app.core.config import get_settings
from app.db.models import Conversation
from app.llm.agents.summary import generate_summary
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatMessage


def build_context_messages(
    *,
    system_prompt: str | None,
    summary: str | None,
    recent_messages: Sequence[ChatMessage],
    current_message: str,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    if summary:
        messages.append(
            ChatMessage(
                role="system",
                content=f"以下是当前会话的摘要记忆，请在回答时参考：\n{summary}",
            )
        )
    messages.extend(recent_messages)
    messages.append(ChatMessage(role="user", content=current_message))
    return messages


def should_refresh_summary(*, total_messages: int, trigger: int) -> bool:
    return total_messages > trigger


async def refresh_summary_if_needed(
    repository: ConversationRepository,
    conversation: Conversation,
) -> None:
    settings = get_settings()
    total_messages = await repository.count_messages(conversation.id)
    if not should_refresh_summary(
        total_messages=total_messages,
        trigger=settings.memory_summary_trigger,
    ):
        return

    cutoff_sequence = total_messages - settings.memory_window_size
    messages_to_summarize = await repository.list_messages_for_summary(
        conversation.id,
        from_sequence_exclusive=conversation.summary_message_count,
        to_sequence_inclusive=cutoff_sequence,
    )
    if not messages_to_summarize:
        return

    summary = await generate_summary(
        previous_summary=conversation.summary,
        messages=[
            ChatMessage(role=message.role, content=message.content)
            for message in messages_to_summarize
        ],
    )
    await repository.update_summary(
        conversation,
        summary=summary,
        summary_message_count=cutoff_sequence,
    )

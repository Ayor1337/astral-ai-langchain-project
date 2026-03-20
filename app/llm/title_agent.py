from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import ModelEndpointSettings, get_settings
from app.llm.base import UpstreamServiceError, create_chat_model, extract_text_content
from app.schemas.chat import ChatMessage

MAX_TITLE_LENGTH = 255


def _resolve_title_agent_config() -> ModelEndpointSettings:
    return get_settings().title_agent_endpoint


def _normalize_title(raw_title: object) -> str:
    if isinstance(raw_title, str):
        normalized = " ".join(raw_title.strip().split())
    else:
        normalized = " ".join(str(raw_title).strip().split())

    if not normalized:
        return "新对话"

    return normalized.splitlines()[0][:MAX_TITLE_LENGTH]


async def generate_conversation_title(messages: Sequence[ChatMessage]) -> str:
    if not messages:
        return "新对话"

    model = create_chat_model(
        endpoint=_resolve_title_agent_config(),
        streaming=False,
    )
    prompt = [
        SystemMessage(
            content=(
                "你是会话标题代理。"
                "请根据给定的首轮问答，为会话生成一个简洁、准确的中文标题。"
                "要求：不超过12个汉字，不要带引号、句号、序号或解释。"
            )
        ),
        HumanMessage(
            content=(
                "首轮问答如下：\n"
                + "\n".join(f"{message.role}: {message.content}" for message in messages)
                + "\n\n请直接输出标题。"
            )
        ),
    ]
    try:
        response = await model.ainvoke(prompt)
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    return _normalize_title(extract_text_content(getattr(response, "content", "")))

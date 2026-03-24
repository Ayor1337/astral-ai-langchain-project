from collections.abc import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.chat import ChatMessage

ContentBlock = dict[str, object]


def _is_mapping(value: object) -> bool:
    return hasattr(value, "items")


def _to_content_block(value: object) -> ContentBlock | None:
    if _is_mapping(value):
        return dict(value)  # type: ignore[arg-type]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if _is_mapping(dumped):
            return dict(dumped)  # type: ignore[arg-type]
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if _is_mapping(dumped):
            return dict(dumped)  # type: ignore[arg-type]
    return None


def normalize_content_blocks(content: object) -> list[ContentBlock]:
    if isinstance(content, str):
        if not content:
            return []
        return [{"type": "text", "text": content, "index": 0}]
    if isinstance(content, (list, tuple)):
        blocks: list[ContentBlock] = []
        for item in content:
            block = _to_content_block(item)
            if block:
                blocks.append(block)
        return blocks
    block = _to_content_block(content)
    if block is not None:
        return [block]
    return []


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content

    text_chunks: list[str] = []
    for block in normalize_content_blocks(content):
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            text_chunks.append(text)
    return "".join(text_chunks)


def to_langchain_message(message: ChatMessage) -> SystemMessage | HumanMessage | AIMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


def to_langchain_messages(messages: Sequence[ChatMessage]) -> list[SystemMessage | HumanMessage | AIMessage]:
    return [to_langchain_message(message) for message in messages]

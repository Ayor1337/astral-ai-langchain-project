from collections.abc import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.chat import ChatMessage

ContentBlock = dict[str, object]


def _is_mapping(value: object) -> bool:
    """判断对象是否可视为字典映射。

    Args:
        value: 待检查对象。

    Returns:
        如果对象可视为映射则返回 True，否则返回 False。
    """
    return hasattr(value, "items")


def _to_content_block(value: object) -> ContentBlock | None:
    """将多种常见对象转换为内容块字典。

    Args:
        value: 待转换对象。

    Returns:
        内容块字典；无法转换时返回 None。
    """
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
    """把模型返回内容标准化为块列表。

    Args:
        content: 原始模型输出内容。

    Returns:
        统一后的内容块列表，便于后续解析文本与结构化事件。
    """
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
    """从混合内容块中提取纯文本。

    Args:
        content: 原始模型输出内容。

    Returns:
        拼接后的纯文本。
    """
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
    """将会话消息映射为 LangChain 消息对象。

    Args:
        message: 领域层消息对象。

    Returns:
        对应角色的 LangChain 消息对象。
    """
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


def to_langchain_messages(messages: Sequence[ChatMessage]) -> list[SystemMessage | HumanMessage | AIMessage]:
    """批量将会话消息转换为 LangChain 消息对象。

    Args:
        messages: 领域层消息列表。

    Returns:
        保持原始顺序的 LangChain 消息列表。
    """
    return [to_langchain_message(message) for message in messages]

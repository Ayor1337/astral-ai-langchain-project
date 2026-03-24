from collections.abc import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.schemas.chat import ChatMessage

ContentBlock = dict[str, object]


def _is_mapping(value: object) -> bool:
    """用最宽松的方式判断对象是否可视为字典。"""
    return hasattr(value, "items")


def _to_content_block(value: object) -> ContentBlock | None:
    """兼容 dict、Pydantic 模型和带 dict/model_dump 的对象。"""
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
    """把模型返回内容标准化为块列表，便于统一处理文本与结构化事件。"""
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
    """从混合内容块中提取纯文本，用于摘要等只关心正文的场景。"""
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
    """按角色映射到 LangChain 消息类型。"""
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


def to_langchain_messages(messages: Sequence[ChatMessage]) -> list[SystemMessage | HumanMessage | AIMessage]:
    """批量转换消息列表，保持原始顺序不变。"""
    return [to_langchain_message(message) for message in messages]

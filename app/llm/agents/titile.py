from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import ConfigurationError, ModelEndpointSettings, get_settings
try:
    from app.llm.messages import extract_text_content
    from app.llm.models.factory import create_chat_model
except ImportError:  # pragma: no cover - compatibility fallback for pre-refactor layout
    from app.llm.base import create_chat_model, extract_text_content
from app.llm.exceptions import UpstreamServiceError

DEFAULT_CONVERSATION_TITLE = "新对话"
TITLE_SYSTEM_PROMPT = (
    "你是会话标题生成器。请根据给定的用户首条消息生成一个简短、准确的标题。"
    "标题必须是单行，不要带引号、前缀、句号或任何解释。"
    "优先跟随用户消息的主要语言。"
)


def create_title_agent(
    *,
    endpoint: ModelEndpointSettings,
):
    """创建专用于标题生成的无工具 agent。

    Args:
        endpoint: 模型端点配置。

    Returns:
        可用于生成会话标题的 LangChain agent。
    """
    model = create_chat_model(
        endpoint=endpoint,
        streaming=False,
        thinking_enabled=False,
    )
    return create_agent(
        model=model,
        tools=[],
        system_prompt=TITLE_SYSTEM_PROMPT,
        name="title_agent",
    )


def _extract_title_text(result: object) -> str:
    """从标题 agent 结果中提取文本。

    Args:
        result: agent 返回结果。

    Returns:
        提取到的标题文本；未命中时返回空字符串。
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, AIMessage):
                    continue
                text = extract_text_content(message.content).strip()
                if text:
                    return text
        return ""

    return extract_text_content(getattr(result, "content", "")).strip()


def _strip_surrounding_quotes(value: str) -> str:
    """去除标题两端可能出现的成对引号。

    Args:
        value: 原始标题文本。

    Returns:
        去除外层引号后的文本。
    """
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
        ("【", "】"),
        ("(", ")"),
        ("（", "）"),
    ]
    stripped = value.strip()
    changed = True
    while changed and stripped:
        changed = False
        for left, right in quote_pairs:
            if stripped.startswith(left) and stripped.endswith(right) and len(stripped) >= 2:
                stripped = stripped[1:-1].strip()
                changed = True
    return stripped


def _normalize_title(raw_title: str) -> str:
    """规范化标题文本并回退默认标题。

    Args:
        raw_title: 模型返回的原始标题。

    Returns:
        规范化后的标题，必要时回退到默认标题。
    """
    normalized = raw_title.strip().splitlines()[0].strip() if raw_title.strip() else ""
    for prefix in ("标题：", "标题:", "title:", "Title:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    normalized = _strip_surrounding_quotes(normalized)
    normalized = normalized.rstrip("。.!！?？:：;；,，、 ").strip()
    return normalized or DEFAULT_CONVERSATION_TITLE


def _resolve_title_agent_endpoint() -> ModelEndpointSettings:
    """解析标题 agent 所使用的模型端点配置。

    Returns:
        标题生成专用的模型端点配置。

    Raises:
        ConfigurationError: 当标题 agent 配置缺失时抛出。
    """
    endpoint = get_settings().title_agent_endpoint
    if endpoint is None:
        raise ConfigurationError("TITLE_AGENT_API_KEY is not configured")
    return endpoint


async def generate_conversation_title(
    *,
    user_message: str,
) -> str:
    """基于用户首条消息生成会话标题。

    Args:
        user_message: 用户的首条消息内容。

    Returns:
        规范化后的会话标题。

    Raises:
        UpstreamServiceError: 当上游标题生成失败时抛出。
    """
    prompt = (
        "请基于以下用户首条消息生成标题。\n\n"
        f"用户：{user_message}\n"
        "\n请直接输出标题。"
    )
    agent = create_title_agent(endpoint=_resolve_title_agent_endpoint())
    try:
        response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    return _normalize_title(_extract_title_text(response))

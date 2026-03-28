from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.base import BaseChatProvider
from app.llm.providers.openai import OpenAIProvider

_PROVIDERS: dict[str, BaseChatProvider] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
}


def get_provider(name: str) -> BaseChatProvider:
    """按名称解析 provider。

    Args:
        name: provider 名称，大小写不敏感。

    Returns:
        匹配到的 provider 实例。

    Raises:
        ValueError: 当名称不受支持时抛出。
    """
    provider = _PROVIDERS.get(name.strip().lower())
    if provider is None:
        raise ValueError(f"unsupported provider: {name}")
    return provider

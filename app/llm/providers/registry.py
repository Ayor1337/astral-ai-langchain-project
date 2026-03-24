from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.base import BaseChatProvider
from app.llm.providers.openai import OpenAIProvider

_PROVIDERS: dict[str, BaseChatProvider] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
}


def get_provider(name: str) -> BaseChatProvider:
    """按名称解析 provider，统一在这里做大小写归一化。"""
    provider = _PROVIDERS.get(name.strip().lower())
    if provider is None:
        raise ValueError(f"unsupported provider: {name}")
    return provider

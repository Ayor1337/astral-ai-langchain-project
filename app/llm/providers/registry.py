from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.base import BaseChatProvider
from app.llm.providers.openai import OpenAIProvider

_PROVIDERS: dict[str, BaseChatProvider] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
}


def get_provider(name: str) -> BaseChatProvider:
    provider = _PROVIDERS.get(name.strip().lower())
    if provider is None:
        raise ValueError(f"unsupported provider: {name}")
    return provider

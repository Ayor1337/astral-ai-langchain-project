from app.core.config import ModelEndpointSettings
from app.llm.capabilities import validate_chat_capabilities
from app.llm.providers import get_provider


def create_chat_model(
    *,
    endpoint: ModelEndpointSettings,
    streaming: bool,
    thinking_enabled: bool = False,
):
    """按 provider 分发到底层模型工厂，隐藏实现差异。"""
    validate_chat_capabilities(
        endpoint=endpoint,
        thinking_enabled=thinking_enabled,
    )
    provider = get_provider(endpoint.provider)
    return provider.create_chat_model(
        endpoint=endpoint,
        streaming=streaming,
        thinking_enabled=thinking_enabled,
    )

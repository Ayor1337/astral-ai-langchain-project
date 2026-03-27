from app.core.config import ModelEndpointSettings
from app.llm.exceptions import ThinkingNotSupportedError
from app.llm.providers import get_provider


def validate_chat_capabilities(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
) -> None:
    """统一校验聊天能力，避免 provider、factory 和 agent 多点判断。"""
    if not thinking_enabled:
        return

    provider = get_provider(endpoint.provider)
    if not provider.supports_thinking:
        raise ThinkingNotSupportedError(f"provider {provider.name} does not support thinking")

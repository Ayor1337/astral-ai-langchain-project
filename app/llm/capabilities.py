from app.core.config import ConfigurationError, ModelEndpointSettings, SearchSettings
from app.llm.exceptions import ThinkingNotSupportedError
from app.llm.providers import get_provider


def validate_chat_capabilities(
    *,
    endpoint: ModelEndpointSettings,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    search: SearchSettings | None = None,
) -> None:
    """统一校验聊天能力，避免 provider、factory 和 agent 多点判断。"""
    if not thinking_enabled:
        if search_enabled:
            _validate_search_settings(search)
        return

    provider = get_provider(endpoint.provider)
    if not provider.supports_thinking:
        raise ThinkingNotSupportedError(f"provider {provider.name} does not support thinking")
    if search_enabled:
        _validate_search_settings(search)


def _validate_search_settings(search: SearchSettings | None) -> None:
    if search is None or not search.api_key.strip():
        raise ConfigurationError("SEARCH_API_KEY is not configured")

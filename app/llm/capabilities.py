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
    """校验聊天请求所需能力是否与 provider 和配置匹配。

    Args:
        endpoint: 模型端点配置。
        thinking_enabled: 是否请求 thinking 能力。
        search_enabled: 是否请求联网搜索能力。
        search: 联网搜索配置。

    Raises:
        ThinkingNotSupportedError: 当 provider 不支持 thinking 时抛出。
        ConfigurationError: 当启用搜索但未提供有效配置时抛出。
    """
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
    """校验联网搜索配置是否可用。

    Args:
        search: 联网搜索配置。

    Raises:
        ConfigurationError: 当 API key 缺失或为空时抛出。
    """
    if search is None or not search.api_key.strip():
        raise ConfigurationError("SEARCH_API_KEY is not configured")

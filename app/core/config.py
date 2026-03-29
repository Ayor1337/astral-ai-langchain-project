from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path

SUPPORTED_PROVIDERS = ("anthropic", "openai")
SUPPORTED_SEARCH_PROVIDERS = ("tavily",)


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class ModelEndpointSettings:
    provider: str
    api_key: str
    base_url: str | None
    model: str


@dataclass(frozen=True)
class SearchSettings:
    provider: str = "tavily"
    api_key: str = ""
    base_url: str = "https://api.tavily.com"
    timeout_seconds: int = 8
    max_results: int = 5


@dataclass(frozen=True)
class AuthSettings:
    jwt_secret_key: str
    jwt_expire_seconds: int = 604800
    jwt_algorithm: str = "HS256"


@dataclass(frozen=True)
class Settings:
    chat_endpoint: ModelEndpointSettings
    title_agent_endpoint: ModelEndpointSettings | None
    database_url: str
    memory_window_size: int
    memory_summary_trigger: int
    search: SearchSettings = field(default_factory=SearchSettings)
    auth: AuthSettings | None = None


def _load_dotenv_values() -> dict[str, str]:
    """读取当前工作目录下的 `.env` 文件。

    Returns:
        按键值对解析后的环境变量回退值。
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values


def _get_setting_value(name: str, default: str = "") -> str:
    """读取配置值，按环境变量、`.env` 和默认值的顺序回退。

    Args:
        name: 配置项名称。
        default: 未命中时使用的默认值。

    Returns:
        解析后的配置字符串。
    """
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value.strip()

    dotenv_value = _load_dotenv_values().get(name)
    if dotenv_value is not None:
        return dotenv_value.strip()

    return default


def _get_int_setting(name: str, default: int) -> int:
    """读取整数配置并在格式错误时失败。

    Args:
        name: 配置项名称。
        default: 未命中时使用的默认值。

    Returns:
        解析后的整数值。

    Raises:
        ConfigurationError: 配置值无法转换为整数时抛出。
    """
    raw_value = _get_setting_value(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _build_endpoint_settings(prefix: str, fallback: ModelEndpointSettings | None = None) -> ModelEndpointSettings:
    """按前缀组装模型端点配置。

    Args:
        prefix: 环境变量前缀。
        fallback: 可选的回退端点配置。

    Returns:
        组装后的端点配置。
    """
    provider = _get_setting_value(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "")
    api_key = _get_setting_value(f"{prefix}_API_KEY") or (fallback.api_key if fallback else "")
    raw_base_url = _get_setting_value(f"{prefix}_BASE_URL")
    base_url = raw_base_url or (fallback.base_url if fallback else None)
    model = _get_setting_value(f"{prefix}_MODEL") or (fallback.model if fallback else "")
    return ModelEndpointSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def _build_optional_endpoint_settings(prefix: str) -> ModelEndpointSettings | None:
    """按前缀构建可选模型端点配置。

    Args:
        prefix: 环境变量前缀。

    Returns:
        启用时返回端点配置，否则返回 `None`。
    """
    raw_values = {
        "provider": _get_setting_value(f"{prefix}_PROVIDER"),
        "api_key": _get_setting_value(f"{prefix}_API_KEY"),
        "base_url": _get_setting_value(f"{prefix}_BASE_URL"),
        "model": _get_setting_value(f"{prefix}_MODEL"),
    }
    if not any(raw_values.values()):
        return None
    return ModelEndpointSettings(
        provider=raw_values["provider"],
        api_key=raw_values["api_key"],
        base_url=raw_values["base_url"] or None,
        model=raw_values["model"],
    )


def _build_search_settings() -> SearchSettings:
    """组装联网搜索配置。

    Returns:
        归一化后的搜索配置。
    """
    return SearchSettings(
        provider=_get_setting_value("SEARCH_PROVIDER", "tavily"),
        api_key=_get_setting_value("SEARCH_API_KEY"),
        base_url=_get_setting_value("SEARCH_BASE_URL", "https://api.tavily.com"),
        timeout_seconds=_get_int_setting("SEARCH_TIMEOUT_SECONDS", 8),
        max_results=_get_int_setting("SEARCH_MAX_RESULTS", 5),
    )


def _build_optional_auth_settings() -> AuthSettings | None:
    """构建可选认证配置。

    Returns:
        存在任意认证配置时返回认证设置，否则返回 `None`。
    """
    raw_secret = _get_setting_value("JWT_SECRET_KEY")
    raw_expire = _get_setting_value("JWT_EXPIRE_SECONDS")
    raw_algorithm = _get_setting_value("JWT_ALGORITHM")
    if not any((raw_secret, raw_expire, raw_algorithm)):
        return None
    return AuthSettings(
        jwt_secret_key=raw_secret,
        jwt_expire_seconds=_get_int_setting("JWT_EXPIRE_SECONDS", 604800),
        jwt_algorithm=raw_algorithm or "HS256",
    )


def _validate_endpoint_settings(prefix: str, endpoint: ModelEndpointSettings) -> ModelEndpointSettings:
    """校验并归一化模型端点配置。

    Args:
        prefix: 环境变量前缀。
        endpoint: 待校验的端点配置。

    Returns:
        归一化后的端点配置。

    Raises:
        ConfigurationError: 端点配置不合法时抛出。
    """
    provider = endpoint.provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"{prefix}_PROVIDER must be one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    api_key = endpoint.api_key.strip()
    if not api_key:
        raise ConfigurationError(f"{prefix}_API_KEY is not configured")

    model = endpoint.model.strip()
    if not model:
        raise ConfigurationError(f"{prefix}_MODEL is not configured")

    base_url = endpoint.base_url.strip() if isinstance(endpoint.base_url, str) else None
    if base_url and not base_url.startswith(("http://", "https://")):
        raise ConfigurationError(f"{prefix}_BASE_URL must start with http:// or https://")

    return ModelEndpointSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url or None,
        model=model,
    )


def _validate_search_settings(search: SearchSettings) -> SearchSettings:
    """校验并归一化搜索配置。

    Args:
        search: 待校验的搜索配置。

    Returns:
        归一化后的搜索配置。

    Raises:
        ConfigurationError: 搜索配置不合法时抛出。
    """
    provider = search.provider.strip().lower()
    if provider not in SUPPORTED_SEARCH_PROVIDERS:
        raise ConfigurationError(
            f"SEARCH_PROVIDER must be one of: {', '.join(SUPPORTED_SEARCH_PROVIDERS)}"
        )

    base_url = search.base_url.strip()
    if not base_url.startswith(("http://", "https://")):
        raise ConfigurationError("SEARCH_BASE_URL must start with http:// or https://")

    if search.timeout_seconds <= 0:
        raise ConfigurationError("SEARCH_TIMEOUT_SECONDS must be greater than 0")

    if search.max_results <= 0:
        raise ConfigurationError("SEARCH_MAX_RESULTS must be greater than 0")

    return SearchSettings(
        provider=provider,
        api_key=search.api_key.strip(),
        base_url=base_url.rstrip("/"),
        timeout_seconds=search.timeout_seconds,
        max_results=search.max_results,
    )


def _validate_auth_settings(auth: AuthSettings | None) -> AuthSettings | None:
    """校验并归一化认证配置。

    Args:
        auth: 待校验的认证配置。

    Returns:
        归一化后的认证配置，未启用时返回 `None`。

    Raises:
        ConfigurationError: 认证配置不合法时抛出。
    """
    if auth is None:
        return None

    secret = auth.jwt_secret_key.strip()
    if not secret:
        raise ConfigurationError("JWT_SECRET_KEY is not configured")

    algorithm = auth.jwt_algorithm.strip().upper()
    if algorithm != "HS256":
        raise ConfigurationError("JWT_ALGORITHM must be HS256")

    if auth.jwt_expire_seconds <= 0:
        raise ConfigurationError("JWT_EXPIRE_SECONDS must be greater than 0")

    return AuthSettings(
        jwt_secret_key=secret,
        jwt_expire_seconds=auth.jwt_expire_seconds,
        jwt_algorithm=algorithm,
    )


def validate_settings(settings: Settings) -> Settings:
    """校验完整配置并返回规范化结果。

    Args:
        settings: 待校验的完整配置对象。

    Returns:
        通过校验后的配置对象。

    Raises:
        ConfigurationError: 任一跨字段约束不满足时抛出。
    """
    chat_endpoint = _validate_endpoint_settings("LLM", settings.chat_endpoint)
    title_agent_endpoint = (
        _validate_endpoint_settings("TITLE_AGENT", settings.title_agent_endpoint)
        if settings.title_agent_endpoint is not None
        else None
    )

    if settings.database_url and not settings.database_url.startswith(
        ("postgresql://", "postgresql+asyncpg://")
    ):
        raise ConfigurationError(
            "DATABASE_URL must start with postgresql:// or postgresql+asyncpg://"
        )

    if settings.memory_window_size <= 0:
        raise ConfigurationError("MEMORY_WINDOW_SIZE must be greater than 0")

    if settings.memory_summary_trigger <= settings.memory_window_size:
        raise ConfigurationError(
            "MEMORY_SUMMARY_TRIGGER must be greater than MEMORY_WINDOW_SIZE"
        )

    return Settings(
        chat_endpoint=chat_endpoint,
        title_agent_endpoint=title_agent_endpoint,
        database_url=settings.database_url,
        memory_window_size=settings.memory_window_size,
        memory_summary_trigger=settings.memory_summary_trigger,
        search=_validate_search_settings(settings.search),
        auth=_validate_auth_settings(settings.auth),
    )


@lru_cache
def get_settings() -> Settings:
    """读取并缓存完整配置对象。

    Returns:
        已校验的配置对象。
    """
    chat_endpoint = _build_endpoint_settings("LLM")
    settings = Settings(
        chat_endpoint=chat_endpoint,
        title_agent_endpoint=_build_optional_endpoint_settings("TITLE_AGENT"),
        database_url=_get_setting_value("DATABASE_URL"),
        memory_window_size=_get_int_setting("MEMORY_WINDOW_SIZE", 8),
        memory_summary_trigger=_get_int_setting("MEMORY_SUMMARY_TRIGGER", 12),
        search=_build_search_settings(),
        auth=_build_optional_auth_settings(),
    )
    return validate_settings(settings)

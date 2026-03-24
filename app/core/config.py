from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

SUPPORTED_PROVIDERS = ("anthropic", "openai")


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class ModelEndpointSettings:
    provider: str
    api_key: str
    base_url: str | None
    model: str


@dataclass(frozen=True)
class Settings:
    chat_endpoint: ModelEndpointSettings
    title_agent_endpoint: ModelEndpointSettings | None
    database_url: str
    memory_window_size: int
    memory_summary_trigger: int


def _load_dotenv_values() -> dict[str, str]:
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
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value.strip()

    dotenv_value = _load_dotenv_values().get(name)
    if dotenv_value is not None:
        return dotenv_value.strip()

    return default


def _get_int_setting(name: str, default: int) -> int:
    raw_value = _get_setting_value(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _build_endpoint_settings(prefix: str, fallback: ModelEndpointSettings | None = None) -> ModelEndpointSettings:
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


def _validate_endpoint_settings(prefix: str, endpoint: ModelEndpointSettings) -> ModelEndpointSettings:
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


def validate_settings(settings: Settings) -> Settings:
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
    )


@lru_cache
def get_settings() -> Settings:
    chat_endpoint = _build_endpoint_settings("LLM")
    settings = Settings(
        chat_endpoint=chat_endpoint,
        title_agent_endpoint=_build_optional_endpoint_settings("TITLE_AGENT"),
        database_url=_get_setting_value("DATABASE_URL"),
        memory_window_size=_get_int_setting("MEMORY_WINDOW_SIZE", 8),
        memory_summary_trigger=_get_int_setting("MEMORY_SUMMARY_TRIGGER", 12),
    )
    return validate_settings(settings)

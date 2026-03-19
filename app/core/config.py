from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    anthropic_base_url: str | None
    anthropic_model: str
    title_agent_model: str | None
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


def validate_settings(settings: Settings) -> Settings:
    if not settings.anthropic_api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured")

    if settings.anthropic_base_url and not settings.anthropic_base_url.startswith(("http://", "https://")):
        if settings.anthropic_model.startswith(("http://", "https://")):
            raise ConfigurationError(
                "ANTHROPIC_BASE_URL looks invalid and ANTHROPIC_MODEL looks like a URL. "
                "Check whether ANTHROPIC_BASE_URL and ANTHROPIC_MODEL are swapped."
            )
        raise ConfigurationError("ANTHROPIC_BASE_URL must start with http:// or https://")

    if not settings.anthropic_model:
        raise ConfigurationError("ANTHROPIC_MODEL is not configured")

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

    return settings


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        anthropic_api_key=_get_setting_value("ANTHROPIC_API_KEY"),
        anthropic_base_url=_get_setting_value("ANTHROPIC_BASE_URL") or None,
        anthropic_model=_get_setting_value("ANTHROPIC_MODEL"),
        title_agent_model=_get_setting_value("TITLE_AGENT_MODEL") or None,
        database_url=_get_setting_value("DATABASE_URL"),
        memory_window_size=_get_int_setting("MEMORY_WINDOW_SIZE", 8),
        memory_summary_trigger=_get_int_setting("MEMORY_SUMMARY_TRIGGER", 12),
    )
    return validate_settings(settings)

from unittest.mock import patch

import pytest

from app.core.config import ModelEndpointSettings
from app.llm.agents.chat import validate_chat_capabilities
from app.llm.exceptions import ThinkingNotSupportedError
from app.llm.models.factory import create_chat_model
from app.llm.providers import get_provider


def test_get_provider_returns_anthropic_provider():
    provider = get_provider("anthropic")

    assert provider.name == "anthropic"
    assert provider.supports_thinking is True


def test_get_provider_returns_openai_provider():
    provider = get_provider("openai")

    assert provider.name == "openai"
    assert provider.supports_thinking is False


def test_get_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unsupported provider"):
        get_provider("unknown")


def test_create_chat_model_uses_openai_provider_for_non_streaming_calls():
    endpoint = ModelEndpointSettings(
        provider="openai",
        api_key="test-key",
        base_url="https://openai.example.com",
        model="gpt-4o-mini",
    )

    with patch("app.llm.providers.openai.ChatOpenAI") as mocked_chat_openai:
        create_chat_model(endpoint=endpoint, streaming=False, thinking_enabled=False)

    assert mocked_chat_openai.call_args.kwargs == {
        "api_key": "test-key",
        "base_url": "https://openai.example.com",
        "model": "gpt-4o-mini",
        "streaming": False,
    }


def test_create_chat_model_rejects_thinking_for_openai():
    endpoint = ModelEndpointSettings(
        provider="openai",
        api_key="test-key",
        base_url=None,
        model="gpt-4o-mini",
    )

    with pytest.raises(ThinkingNotSupportedError, match="provider openai does not support thinking"):
        create_chat_model(endpoint=endpoint, streaming=True, thinking_enabled=True)


def test_validate_chat_capabilities_rejects_thinking_for_openai():
    endpoint = ModelEndpointSettings(
        provider="openai",
        api_key="test-key",
        base_url=None,
        model="gpt-4o-mini",
    )

    with pytest.raises(ThinkingNotSupportedError, match="provider openai does not support thinking"):
        validate_chat_capabilities(endpoint=endpoint, thinking_enabled=True)

from typing import Any, Literal, Protocol

from app.core.config import ModelEndpointSettings

ProviderName = Literal["anthropic", "openai"]


class BaseChatProvider(Protocol):
    name: ProviderName
    supports_thinking: bool

    def validate_chat_capabilities(
        self,
        *,
        endpoint: ModelEndpointSettings,
        thinking_enabled: bool = False,
    ) -> None: ...

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> Any: ...

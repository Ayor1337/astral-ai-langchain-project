from langchain_openai import ChatOpenAI

from app.core.config import ModelEndpointSettings
from app.llm.exceptions import ThinkingNotSupportedError


class OpenAIProvider:
    name = "openai"
    supports_thinking = False

    def validate_chat_capabilities(
        self,
        *,
        endpoint: ModelEndpointSettings,
        thinking_enabled: bool = False,
    ) -> None:
        if thinking_enabled:
            raise ThinkingNotSupportedError("provider openai does not support thinking")

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> ChatOpenAI:
        self.validate_chat_capabilities(
            endpoint=endpoint,
            thinking_enabled=thinking_enabled,
        )

        return ChatOpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
        )

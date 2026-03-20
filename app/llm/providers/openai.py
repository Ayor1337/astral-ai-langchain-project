from langchain_openai import ChatOpenAI

from app.core.config import ModelEndpointSettings
from app.llm.exceptions import ThinkingNotSupportedError


class OpenAIProvider:
    name = "openai"
    supports_thinking = False

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> ChatOpenAI:
        if thinking_enabled:
            raise ThinkingNotSupportedError("provider openai does not support thinking")

        return ChatOpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
        )

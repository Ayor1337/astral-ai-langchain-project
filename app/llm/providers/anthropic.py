from langchain_anthropic import ChatAnthropic

from app.core.config import ModelEndpointSettings


def _disabled_thinking() -> dict[str, str]:
    return {"type": "disabled"}


def _adaptive_thinking() -> dict[str, str]:
    return {"type": "adaptive", "display": "summarized"}


class AnthropicProvider:
    name = "anthropic"
    supports_thinking = True

    def validate_chat_capabilities(
        self,
        *,
        endpoint: ModelEndpointSettings,
        thinking_enabled: bool = False,
    ) -> None:
        return None

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> ChatAnthropic:
        return ChatAnthropic(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
            temperature=0.7,
            thinking=_adaptive_thinking() if thinking_enabled else _disabled_thinking(),
        )

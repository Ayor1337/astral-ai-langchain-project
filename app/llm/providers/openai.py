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
        """OpenAI 兼容实现当前不支持 thinking，提早在入口处拒绝。"""
        if thinking_enabled:
            raise ThinkingNotSupportedError("provider openai does not support thinking")

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> ChatOpenAI:
        """构造 LangChain OpenAI 聊天模型实例。"""
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

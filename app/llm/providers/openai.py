from langchain_openai import ChatOpenAI

from app.core.config import ModelEndpointSettings


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
        """构造 LangChain OpenAI 聊天模型实例。"""
        return ChatOpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
        )

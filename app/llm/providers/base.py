from typing import Any, Literal, Protocol

from app.core.config import ModelEndpointSettings

ProviderName = Literal["anthropic", "openai"]


class BaseChatProvider(Protocol):
    name: ProviderName
    supports_thinking: bool

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> Any:
        """创建聊天模型实例。

        Args:
            endpoint: 模型端点配置。
            streaming: 是否启用流式输出。
            thinking_enabled: 是否请求 thinking 能力。

        Returns:
            任意兼容的 LangChain 聊天模型对象。
        """
        ...

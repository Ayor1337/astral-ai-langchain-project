from langchain_anthropic import ChatAnthropic

from app.core.config import ModelEndpointSettings


def _disabled_thinking() -> dict[str, str]:
    """构造关闭 thinking 的 Anthropic 配置。

    Returns:
        关闭 thinking 的配置字典。
    """
    return {"type": "disabled"}


def _adaptive_thinking() -> dict[str, str]:
    """构造启用自适应 thinking 的 Anthropic 配置。

    Returns:
        启用自适应 thinking 的配置字典。
    """
    return {"type": "adaptive", "display": "summarized"}


class AnthropicProvider:
    name = "anthropic"
    supports_thinking = True

    def create_chat_model(
        self,
        *,
        endpoint: ModelEndpointSettings,
        streaming: bool,
        thinking_enabled: bool = False,
    ) -> ChatAnthropic:
        """构造 Anthropic 聊天模型，并根据开关切换 thinking 模式。

        Args:
            endpoint: 模型端点配置。
            streaming: 是否启用流式输出。
            thinking_enabled: 是否启用 thinking。

        Returns:
            已配置好的 ChatAnthropic 实例。
        """
        return ChatAnthropic(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
            temperature=0.7,
            thinking=_adaptive_thinking() if thinking_enabled else _disabled_thinking(),
        )

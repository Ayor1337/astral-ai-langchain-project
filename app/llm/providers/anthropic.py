from langchain_anthropic import ChatAnthropic

from app.core.config import ModelEndpointSettings


def _disabled_thinking() -> dict[str, str]:
    """显式关闭 thinking，保持传给模型的配置结构稳定。"""
    return {"type": "disabled"}


def _adaptive_thinking() -> dict[str, str]:
    """启用 Anthropic 的自适应 thinking，并要求返回摘要化展示。"""
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
        """构造 Anthropic 聊天模型，并根据开关切换 thinking 模式。"""
        return ChatAnthropic(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            model=endpoint.model,
            streaming=streaming,
            temperature=0.7,
            thinking=_adaptive_thinking() if thinking_enabled else _disabled_thinking(),
        )

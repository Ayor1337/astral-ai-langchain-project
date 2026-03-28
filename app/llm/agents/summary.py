from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import ModelEndpointSettings, get_settings
from app.llm.exceptions import UpstreamServiceError
from app.llm.messages import extract_text_content
from app.llm.models.factory import create_chat_model
from app.schemas.chat import ChatMessage

SUMMARY_SYSTEM_PROMPT = (
    "你是对话记忆压缩器。请把新增对话压缩成简洁摘要，"
    "保留用户目标、约束、已完成事项和待办，不要编造内容。"
)


def create_summary_agent(
    *,
    endpoint: ModelEndpointSettings,
):
    """创建专用于摘要压缩的无工具 agent。

    Args:
        endpoint: 模型端点配置。

    Returns:
        可用于生成摘要的 LangChain agent。
    """
    model = create_chat_model(
        endpoint=endpoint,
        streaming=False,
        thinking_enabled=False,
    )
    return create_agent(
        model=model,
        tools=[],
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        name="summary_agent",
    )


def _extract_summary_text(result: object) -> str:
    """从 agent 结果中提取摘要文本。

    Args:
        result: agent 返回结果。

    Returns:
        提取到的摘要文本；未命中时返回空字符串。
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, AIMessage):
                    continue
                text = extract_text_content(message.content).strip()
                if text:
                    return text
        return ""

    return extract_text_content(getattr(result, "content", "")).strip()


async def generate_summary(
    *,
    previous_summary: str | None,
    messages: Sequence[ChatMessage],
) -> str:
    """基于旧摘要和新增消息生成新的摘要文本。

    Args:
        previous_summary: 现有摘要。
        messages: 需要压缩的新消息序列。

    Returns:
        更新后的摘要文本。
    """
    if not messages:
        return previous_summary or ""

    prompt = (
        f"已有摘要：\n{previous_summary or '无'}\n\n"
        "新增消息：\n"
        + "\n".join(f"{message.role}: {message.content}" for message in messages)
        + "\n\n请输出更新后的摘要。"
    )
    agent = create_summary_agent(endpoint=get_settings().chat_endpoint)
    try:
        response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    return _extract_summary_text(response)

import json
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from app.core.config import ConfigurationError, get_settings
from app.llm.base import UpstreamServiceError, disabled_thinking, extract_text_content

MAX_REASONING_SUMMARY_LENGTH = 500
REASONING_CHUNK_SIZE = 24
MAX_THOUGHT_TITLE_LENGTH = 32
MAX_THOUGHT_MESSAGE_LENGTH = 120


def _resolve_reasoning_agent_config() -> tuple[str, str | None, str]:
    settings = get_settings()
    api_key = settings.anthropic_api_key
    base_url = settings.anthropic_base_url
    model = settings.title_agent_model or settings.anthropic_model
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured")
    return api_key, base_url, model


def _normalize_summary(raw_summary: object) -> str:
    if isinstance(raw_summary, str):
        normalized = " ".join(raw_summary.strip().split())
    else:
        normalized = " ".join(str(raw_summary).strip().split())
    if not normalized:
        return ""
    return normalized[:MAX_REASONING_SUMMARY_LENGTH]


def _split_summary(summary: str) -> list[str]:
    if not summary:
        return []
    return [summary[i : i + REASONING_CHUNK_SIZE] for i in range(0, len(summary), REASONING_CHUNK_SIZE)]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _normalize_thought_text(raw_value: object, *, limit: int) -> str:
    if isinstance(raw_value, str):
        normalized = " ".join(raw_value.strip().split())
    else:
        normalized = " ".join(str(raw_value).strip().split())
    if not normalized:
        return ""
    return normalized[:limit]


def _normalize_thought_steps(raw_steps: object) -> list[dict[str, str]]:
    if not isinstance(raw_steps, list):
        return []

    normalized_steps: list[dict[str, str]] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        title = _normalize_thought_text(item.get("title") or item.get("name") or "", limit=MAX_THOUGHT_TITLE_LENGTH)
        message = _normalize_thought_text(
            item.get("message") or item.get("content") or item.get("summary") or "",
            limit=MAX_THOUGHT_MESSAGE_LENGTH,
        )
        if not message:
            continue
        normalized_steps.append(
            {
                "title": title or f"思考步骤 {len(normalized_steps) + 1}",
                "message": message,
            }
        )
    return normalized_steps


def _parse_thought_steps(raw_text: str) -> list[dict[str, str]]:
    cleaned = _strip_code_fence(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise UpstreamServiceError("invalid thought steps payload") from exc

    steps = _normalize_thought_steps(parsed)
    if not steps:
        raise UpstreamServiceError("empty thought steps payload")
    return steps


async def generate_reasoning_summary(*, user_message: str, assistant_message: str) -> str:
    if not user_message.strip() or not assistant_message.strip():
        return ""

    api_key, base_url, model_name = _resolve_reasoning_agent_config()
    model = ChatAnthropic(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        streaming=False,
        thinking=disabled_thinking(),
    )
    prompt = [
        SystemMessage(
            content=(
                "你是对话思路摘要器。"
                "请基于用户问题与助手回答，输出简洁的“思路摘要”。"
                "必须是可对外展示的高层说明，不能包含链路思维细节、隐含推理步骤或自我反思。"
                "限制在1-2句中文。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：{user_message}\n"
                f"助手回答：{assistant_message}\n\n"
                "请直接输出思路摘要，不要加前缀。"
            )
        ),
    ]
    try:
        response = await model.ainvoke(prompt)
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    return _normalize_summary(extract_text_content(getattr(response, "content", "")))


async def generate_thought_steps(
    *,
    user_message: str,
    raw_thinking: str,
    existing_steps: list[dict[str, object]] | None = None,
) -> list[dict[str, str]]:
    if not user_message.strip() or not raw_thinking.strip():
        return []

    api_key, base_url, model_name = _resolve_reasoning_agent_config()
    model = ChatAnthropic(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        streaming=False,
        thinking=disabled_thinking(),
    )
    existing_steps_json = json.dumps(existing_steps or [], ensure_ascii=False, separators=(",", ":"))
    prompt = [
        SystemMessage(
            content=(
                "你是思考步骤整理器。"
                "请把原始模型 thinking 改写为适合前端逐步展示的中文步骤列表。"
                "每一步只保留一个明确动作或意图，避免空话、重复和自我反思。"
                "步数应根据问题复杂度动态决定，不要强行凑固定数量。"
                "输出必须是 JSON 数组，数组项格式为"
                '{"title":"步骤标题","message":"步骤说明"}。'
                "不要输出 markdown、解释或多余前缀。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：{user_message}\n"
                f"原始 thinking：{raw_thinking}\n"
                f"已展示步骤：{existing_steps_json}\n\n"
                "请返回完整步骤列表。"
            )
        ),
    ]
    try:
        response = await model.ainvoke(prompt)
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    raw_text = extract_text_content(getattr(response, "content", ""))
    return _parse_thought_steps(raw_text)


async def stream_reasoning_chunks(*, user_message: str, assistant_message: str) -> AsyncIterator[str]:
    summary = await generate_reasoning_summary(
        user_message=user_message,
        assistant_message=assistant_message,
    )
    for chunk in _split_summary(summary):
        yield chunk

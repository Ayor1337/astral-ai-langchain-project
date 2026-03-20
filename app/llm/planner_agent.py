from __future__ import annotations

import json
from json import JSONDecodeError

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import ConfigurationError, get_settings
from app.llm.base import UpstreamServiceError, disabled_thinking, extract_text_content

PLANNER_TOOL_WHITELIST = ("web_search", "http_fetch")
PLANNER_MODEL_NAME = "MiniMax-M2.1-highspeed"


def _resolve_planner_agent_config() -> tuple[str, str | None, str]:
    settings = get_settings()
    api_key = settings.anthropic_api_key
    base_url = settings.anthropic_base_url
    model = PLANNER_MODEL_NAME
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured")
    return api_key, base_url, model


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        raise ValueError("planner returned empty output")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("planner output does not contain JSON")
    decoder = json.JSONDecoder()
    parsed, end_index = decoder.raw_decode(text[start:])
    return json.dumps(parsed, ensure_ascii=False)


def _normalize_plan(raw_plan: object) -> list[str]:
    if not isinstance(raw_plan, list):
        raise ValueError("planner plan must be a list")
    normalized = [str(item).strip() for item in raw_plan if str(item).strip()]
    if not normalized:
        raise ValueError("planner plan must not be empty")
    return normalized


def _normalize_tools(raw_tools: object) -> list[str]:
    if not isinstance(raw_tools, list):
        raise ValueError("planner tools must be a list")
    normalized = [str(item).strip() for item in raw_tools if str(item).strip()]
    if not normalized:
        raise ValueError("planner tools must not be empty")
    invalid_tools = [tool for tool in normalized if tool not in PLANNER_TOOL_WHITELIST]
    if invalid_tools:
        raise ValueError(f"planner returned unsupported tools: {', '.join(invalid_tools)}")
    return normalized


def _normalize_route(raw_payload: object) -> dict[str, object]:
    if not isinstance(raw_payload, dict):
        raise ValueError("planner output must be a JSON object")

    route = str(raw_payload.get("route", "")).strip()
    if route == "complex_with_tools":
        route = "agent"

    if route == "simple":
        return {"route": "simple"}

    if route == "complex":
        return {
            "route": "complex",
            "plan": _normalize_plan(raw_payload.get("plan")),
        }

    if route == "agent":
        return {
            "route": "agent",
            "plan": _normalize_plan(raw_payload.get("plan")),
            "tools": _normalize_tools(raw_payload.get("tools")),
        }

    raise ValueError(f"planner returned unsupported route: {route}")


async def plan_execution_route(*, message: str) -> dict[str, object]:
    if not message.strip():
        raise ValueError("planner message must not be empty")

    api_key, base_url, model_name = _resolve_planner_agent_config()
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
                "You are the reasoning and execution planner for an AI chat system.\n"
                "Your job is NOT to answer the user directly.\n"
                "Classify the request into one of: simple, complex, agent.\n"
                "If route is simple, return exactly {\"route\":\"simple\"}.\n"
                "If route is complex, return exactly {\"route\":\"complex\",\"plan\":[...]}.\n"
                "If route is agent, return exactly "
                "{\"route\":\"agent\",\"plan\":[...],\"tools\":[...]}. "
                f"Allowed tools: {', '.join(PLANNER_TOOL_WHITELIST)}.\n"
                "Always return valid JSON only. Do not wrap JSON in markdown. "
                "Do not include any explanation or extra keys."
            )
        ),
        HumanMessage(content=message),
    ]
    try:
        response = await model.ainvoke(prompt)
    except Exception as exc:
        raise UpstreamServiceError(str(exc)) from exc

    raw_text = extract_text_content(getattr(response, "content", ""))
    try:
        normalized_json = _extract_json_text(raw_text)
        parsed = json.loads(normalized_json)
        return _normalize_route(parsed)
    except (JSONDecodeError, ValueError) as exc:
        raise UpstreamServiceError(str(exc)) from exc

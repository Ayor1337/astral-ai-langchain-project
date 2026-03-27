from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import tool


def add(a: int, b: int) -> dict[str, int]:
    """Add two integers and return a JSON-serializable result."""
    return {"result": a + b}


def get_chat_tools(*, search_fn: Callable[[str], Awaitable[dict[str, Any]]] | None = None) -> list:
    tools: list = [add]
    if search_fn is None:
        return tools

    @tool
    async def web_search(query: str) -> dict[str, Any]:
        """Search the web for current information and return structured results."""
        return await search_fn(query)

    tools.append(web_search)
    return tools

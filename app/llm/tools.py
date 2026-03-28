from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import tool


def add(a: int, b: int) -> dict[str, int]:
    """把两个整数相加，并返回可 JSON 序列化的结果。

    Args:
        a: 第一个整数。
        b: 第二个整数。

    Returns:
        包含加和结果的字典。
    """
    return {"result": a + b}


def get_chat_tools(*, search_fn: Callable[[str], Awaitable[dict[str, Any]]] | None = None) -> list:
    """构造聊天 agent 使用的工具列表。

    Args:
        search_fn: 可选的联网搜索函数，传入后会附加 web_search 工具。

    Returns:
        可直接传给 agent 的工具列表。
    """
    tools: list = [add]
    if search_fn is None:
        return tools

    @tool
    async def web_search(query: str) -> dict[str, Any]:
        """调用外部搜索函数并返回结构化结果。

        Args:
            query: 搜索关键词。

        Returns:
            联网搜索结果字典。
        """
        return await search_fn(query)

    tools.append(web_search)
    return tools

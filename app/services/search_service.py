from __future__ import annotations

import inspect
from typing import Any

import httpx

from app.core.config import SearchSettings


class TavilySearchService:
    """封装 Tavily 搜索，并把结果收敛为对模型稳定的最小结构。

    对上层只暴露稳定的查询与结果形状，不泄漏 HTTP 细节。
    """

    def __init__(self, settings: SearchSettings):
        """初始化 Tavily 搜索服务。

        Args:
            self: 服务实例本身。
            settings: 搜索配置。
        """
        self._settings = settings

    async def search(self, query: str) -> dict[str, object]:
        """执行 Tavily 搜索并归一化返回结构。

        请求失败或超时时返回空结果和错误信息，保证上层响应稳定。
        """
        try:
            async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
                response = await client.post(
                    f"{self._settings.base_url}/search",
                    json={
                        "api_key": self._settings.api_key,
                        "query": query,
                        "max_results": self._settings.max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                raise_result = response.raise_for_status()
                if inspect.isawaitable(raise_result):
                    await raise_result
                payload = response.json()
                if inspect.isawaitable(payload):
                    payload = await payload
        except httpx.TimeoutException:
            return {"query": query, "results": [], "error": "search request timed out"}
        except httpx.HTTPError as exc:
            return {"query": query, "results": [], "error": f"search request failed: {exc}"}
        except ValueError:
            return {"query": query, "results": [], "error": "search response was invalid"}

        return {
            "query": query,
            "results": _normalize_search_results(payload),
        }


def _normalize_search_results(payload: object) -> list[dict[str, str]]:
    """从 Tavily 响应中提取稳定的搜索结果列表。

    只保留标题、链接和摘要，过滤掉无效条目。
    """
    if not isinstance(payload, dict):
        return []

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content") or item.get("snippet") or item.get("description") or ""
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(snippet, str):
            snippet = ""
        results.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "snippet": snippet.strip(),
            }
        )
    return results

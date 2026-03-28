from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import SearchSettings
from app.services.search_service import TavilySearchService


@pytest.mark.anyio
async def test_tavily_search_service_normalizes_results():
    payload = {
        "results": [
            {
                "title": "Astral AI",
                "url": "https://example.com/astral",
                "content": "Astral AI latest update",
                "score": 0.95,
            },
            {
                "title": "Ignored missing url",
                "content": "Should be filtered out",
            },
        ]
    }

    response = AsyncMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    with patch("app.services.search_service.httpx.AsyncClient.post", new=AsyncMock(return_value=response)):
        service = TavilySearchService(
            SearchSettings(
                provider="tavily",
                api_key="search-key",
                base_url="https://api.tavily.com",
                timeout_seconds=8,
                max_results=5,
            )
        )
        result = await service.search("Astral AI 最新消息")

    assert result == {
        "query": "Astral AI 最新消息",
        "results": [
            {
                "title": "Astral AI",
                "url": "https://example.com/astral",
                "snippet": "Astral AI latest update",
            }
        ],
    }


@pytest.mark.anyio
async def test_tavily_search_service_degrades_on_timeout():
    with patch(
        "app.services.search_service.httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.TimeoutException("timed out")),
    ):
        service = TavilySearchService(
            SearchSettings(
                provider="tavily",
                api_key="search-key",
                base_url="https://api.tavily.com",
                timeout_seconds=8,
                max_results=5,
            )
        )
        result = await service.search("Astral AI 最新消息")

    assert result["query"] == "Astral AI 最新消息"
    assert result["results"] == []
    assert result["error"] == "search request timed out"


@pytest.mark.anyio
async def test_tavily_search_service_degrades_when_response_json_is_invalid():
    response = AsyncMock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("invalid json")

    with patch("app.services.search_service.httpx.AsyncClient.post", new=AsyncMock(return_value=response)):
        service = TavilySearchService(
            SearchSettings(
                provider="tavily",
                api_key="search-key",
                base_url="https://api.tavily.com",
                timeout_seconds=8,
                max_results=5,
            )
        )
        result = await service.search("Astral AI 最新消息")

    assert result["query"] == "Astral AI 最新消息"
    assert result["results"] == []
    assert result["error"] == "search response was invalid"

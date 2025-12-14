"""SearXNG search client library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .result import Err, Ok, Result
from .types import Infobox, InfoboxUrl, SearchResponse, SearchResult, SearxngError

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 30.0


def _parse_search_result(data: dict[str, Any]) -> SearchResult:
    """Parse a single search result from API response."""
    return SearchResult(
        url=data.get("url", ""),
        title=data.get("title", ""),
        content=data.get("content", ""),
    )


def _parse_infobox(data: dict[str, Any]) -> Infobox:
    """Parse an infobox from API response."""
    urls = [
        InfoboxUrl(title=u.get("title", ""), url=u.get("url", ""))
        for u in data.get("urls", [])
    ]
    return Infobox(
        infobox=data.get("infobox", ""),
        id=data.get("id", ""),
        content=data.get("content", ""),
        urls=urls,
    )


def _parse_response(data: dict[str, Any]) -> SearchResponse:
    """Parse full search response from API."""
    results = [_parse_search_result(r) for r in data.get("results", [])]
    infoboxes = [_parse_infobox(ib) for ib in data.get("infoboxes", [])]
    return SearchResponse(
        query=data.get("query", ""),
        number_of_results=data.get("number_of_results", len(results)),
        results=results,
        infoboxes=infoboxes,
    )


def search(
    query: str,
    limit: int = 10,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result[SearchResponse, SearxngError]:
    """Search using SearXNG instance.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        base_url: Base URL of SearXNG instance (default: http://localhost:8080)
        timeout: Request timeout in seconds

    Returns:
        Result containing SearchResponse on success or SearxngError on failure
    """
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            params = {"q": query, "format": "json"}
            response = client.get("/search", params=params)
            response.raise_for_status()

        data = response.json()
        parsed = _parse_response(data)

        # Limit results
        limited_results = parsed.results[:limit]
        return Ok(
            SearchResponse(
                query=parsed.query,
                number_of_results=parsed.number_of_results,
                results=limited_results,
                infoboxes=parsed.infoboxes,
            )
        )

    except httpx.TimeoutException:
        return Err(SearxngError("Request timed out"))
    except httpx.HTTPStatusError as e:
        return Err(SearxngError(f"HTTP error: {e.response.status_code}", e.response.status_code))
    except httpx.RequestError as e:
        return Err(SearxngError(f"Request failed: {e}"))
    except json.JSONDecodeError:
        return Err(SearxngError("Invalid JSON response"))


async def search_async(
    query: str,
    limit: int = 10,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result[SearchResponse, SearxngError]:
    """Search using SearXNG instance (async version).

    Args:
        query: Search query string
        limit: Maximum number of results to return
        base_url: Base URL of SearXNG instance (default: http://localhost:8080)
        timeout: Request timeout in seconds

    Returns:
        Result containing SearchResponse on success or SearxngError on failure
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            params = {"q": query, "format": "json"}
            response = await client.get("/search", params=params)
            response.raise_for_status()

        data = response.json()
        parsed = _parse_response(data)

        # Limit results
        limited_results = parsed.results[:limit]
        return Ok(
            SearchResponse(
                query=parsed.query,
                number_of_results=parsed.number_of_results,
                results=limited_results,
                infoboxes=parsed.infoboxes,
            )
        )

    except httpx.TimeoutException:
        return Err(SearxngError("Request timed out"))
    except httpx.HTTPStatusError as e:
        return Err(SearxngError(f"HTTP error: {e.response.status_code}", e.response.status_code))
    except httpx.RequestError as e:
        return Err(SearxngError(f"Request failed: {e}"))
    except json.JSONDecodeError:
        return Err(SearxngError("Invalid JSON response"))


@dataclass
class SearxngClient:
    """SearXNG search client.

    Usage:
        client = SearxngClient(base_url="http://localhost:8080")

        # Sync search
        result = client.search("python async frameworks")
        if result.is_ok():
            for r in result.value.results:
                print(f"{r.title}: {r.url}")

        # Async search
        result = await client.search_async("python async frameworks")
    """

    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT

    def search(self, query: str, limit: int = 10) -> Result[SearchResponse, SearxngError]:
        """Search using SearXNG (sync)."""
        return search(query, limit, self.base_url, self.timeout)

    async def search_async(
        self, query: str, limit: int = 10
    ) -> Result[SearchResponse, SearxngError]:
        """Search using SearXNG (async)."""
        return await search_async(query, limit, self.base_url, self.timeout)

    def format_results(self, response: SearchResponse) -> str:
        """Format search results as readable text."""
        text = ""

        for infobox in response.infoboxes:
            text += f"Infobox: {infobox.infobox}\n"
            text += f"ID: {infobox.id}\n"
            text += f"Content: {infobox.content}\n"
            text += "\n"

        if not response.results:
            text += "No results found\n"
        else:
            for result in response.results:
                text += f"Title: {result.title}\n"
                text += f"URL: {result.url}\n"
                text += f"Content: {result.content}\n"
                text += "\n"

        return text.rstrip()

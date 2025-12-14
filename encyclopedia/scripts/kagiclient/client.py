"""Kagi API client library."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import httpx

from .result import Err, Ok, Result
from .types import (
    ErrorCode,
    KagiError,
    SearchResponse,
    SearchResult,
    SummarizerEngine,
    SummaryResponse,
    SummaryType,
)

KAGI_API_BASE = "https://kagi.com/api/v0"
DEFAULT_TIMEOUT = 30.0


def _get_api_key(api_key: str | None = None) -> str | None:
    """Get API key from parameter or environment."""
    return api_key or os.environ.get("KAGI_API_KEY")


@dataclass
class KagiClient:
    """Kagi API client for search and summarization.

    Usage:
        client = KagiClient(api_key="your-key")  # or set KAGI_API_KEY env var

        # Search
        result = await client.search("python async frameworks")
        if result.is_ok():
            for r in result.value.results:
                print(f"{r.title}: {r.url}")

        # Summarize
        result = await client.summarize("https://example.com/article")
        if result.is_ok():
            print(result.value.summary)
    """

    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._resolved_key = _get_api_key(self.api_key)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bot {self._resolved_key}",
            "Content-Type": "application/json",
        }

    def _check_key(self) -> Result[None, KagiError]:
        """Verify API key is available."""
        if not self._resolved_key:
            return Err(KagiError(ErrorCode.CONFIG_ERROR, "KAGI_API_KEY not configured"))
        return Ok(None)

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> Result[SearchResponse, KagiError]:
        """Search the web using Kagi Search API.

        Args:
            query: Search query string
            limit: Maximum number of results (default: 10)

        Returns:
            Result containing SearchResponse on success or KagiError on failure
        """
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{KAGI_API_BASE}/search",
                    params={"q": query, "limit": limit},
                    headers=self._headers(),
                )

                if resp.status_code == 401:
                    return Err(KagiError(ErrorCode.CONFIG_ERROR, "Invalid KAGI_API_KEY"))
                if resp.status_code == 402:
                    return Err(KagiError(ErrorCode.CONFIG_ERROR, "Kagi API credits exhausted"))
                if resp.status_code == 429:
                    return Err(KagiError(ErrorCode.BACKEND_ERROR, "Rate limit exceeded"))
                if resp.status_code != 200:
                    return Err(
                        KagiError(ErrorCode.BACKEND_ERROR, f"Kagi API error: {resp.status_code}")
                    )

                data = resp.json()

                # Parse results - filter for search results (t == 0)
                results = []
                for item in data.get("data", []):
                    if item.get("t") == 0:  # Search result type
                        results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                snippet=item.get("snippet", ""),
                                published=item.get("published"),
                            )
                        )

                return Ok(
                    SearchResponse(
                        query=query,
                        results=results[:limit],
                        result_count=len(results),
                    )
                )

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))

    async def summarize(
        self,
        url: str,
        summary_type: Literal["summary", "takeaway"] | SummaryType = SummaryType.SUMMARY,
        engine: Literal["cecil", "agnes", "daphne", "muriel"] | SummarizerEngine = SummarizerEngine.CECIL,
        target_language: str | None = None,
    ) -> Result[SummaryResponse, KagiError]:
        """Summarize content from a URL.

        Args:
            url: URL to summarize
            summary_type: "summary" for prose, "takeaway" for bullet points
            engine: Summarizer engine to use
            target_language: Optional target language code (e.g., "EN", "DE")

        Returns:
            Result containing SummaryResponse on success or KagiError on failure
        """
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        # Normalize enum values
        if isinstance(summary_type, str):
            summary_type = SummaryType(summary_type)
        if isinstance(engine, str):
            engine = SummarizerEngine(engine)

        payload: dict = {
            "url": url,
            "summary_type": summary_type.value,
            "engine": engine.value,
        }
        if target_language:
            payload["target_language"] = target_language

        try:
            # Summarization can take longer
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                resp = await client.post(
                    f"{KAGI_API_BASE}/summarize",
                    json=payload,
                    headers=self._headers(),
                )

                if resp.status_code == 401:
                    return Err(KagiError(ErrorCode.CONFIG_ERROR, "Invalid KAGI_API_KEY"))
                if resp.status_code == 402:
                    return Err(KagiError(ErrorCode.CONFIG_ERROR, "Kagi API credits exhausted"))
                if resp.status_code != 200:
                    return Err(
                        KagiError(ErrorCode.BACKEND_ERROR, f"Kagi API error: {resp.status_code}")
                    )

                data = resp.json()
                output = data.get("data", {}).get("output", "")

                return Ok(
                    SummaryResponse(
                        url=url,
                        summary=output,
                        summary_type=summary_type,
                        engine=engine,
                    )
                )

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))


# Convenience functions for one-off usage
async def search(
    query: str,
    limit: int = 10,
    api_key: str | None = None,
) -> Result[SearchResponse, KagiError]:
    """Search the web using Kagi Search API.

    Convenience function that creates a client for single use.
    For multiple requests, prefer creating a KagiClient instance.
    """
    client = KagiClient(api_key=api_key)
    return await client.search(query, limit)


async def summarize(
    url: str,
    summary_type: Literal["summary", "takeaway"] = "summary",
    engine: Literal["cecil", "agnes", "daphne", "muriel"] = "cecil",
    target_language: str | None = None,
    api_key: str | None = None,
) -> Result[SummaryResponse, KagiError]:
    """Summarize content from a URL.

    Convenience function that creates a client for single use.
    For multiple requests, prefer creating a KagiClient instance.
    """
    client = KagiClient(api_key=api_key)
    return await client.summarize(url, summary_type, engine, target_language)

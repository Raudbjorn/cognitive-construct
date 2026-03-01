"""Kagi API client library.

Wraps the Kagi v0 API (search, summarize, fastgpt, enrich) using httpx.
API reference: https://help.kagi.com/kagi/api/overview.html
Implementation based on: https://github.com/kagisearch/kagiapi
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .result import Err, Ok, Result
from .types import (
    EnrichResponse,
    EnrichResult,
    ErrorCode,
    FastGPTReference,
    FastGPTResponse,
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


def _parse_error(resp: httpx.Response) -> KagiError:
    """Extract error message from Kagi API response."""
    try:
        errors = resp.json().get("error", [])
        msg = errors[0].get("msg", resp.reason_phrase) if errors else resp.reason_phrase
    except Exception:
        msg = resp.reason_phrase or f"HTTP {resp.status_code}"
    return KagiError(ErrorCode.BACKEND_ERROR, msg)


def _check_status(resp: httpx.Response) -> KagiError | None:
    """Check response status and return error if not 200."""
    if resp.status_code == 200:
        return None
    if resp.status_code == 401:
        err = _parse_error(resp)
        return KagiError(ErrorCode.CONFIG_ERROR, err.message)
    if resp.status_code == 402:
        return KagiError(ErrorCode.CONFIG_ERROR, "Kagi API credits exhausted")
    if resp.status_code == 429:
        return KagiError(ErrorCode.BACKEND_ERROR, "Rate limit exceeded")
    return _parse_error(resp)


@dataclass
class KagiClient:
    """Kagi API client for search, summarization, FastGPT, and enrich.

    Usage:
        client = KagiClient(api_key="your-key")  # or set KAGI_API_KEY env var

        # FastGPT (AI answer with references — no beta required)
        result = await client.fastgpt("explain ownership in Rust")
        if result.is_ok():
            print(result.value.output)
            for ref in result.value.references:
                print(f"  {ref.title}: {ref.url}")

        # Search (requires Search API beta access)
        result = await client.search("python async frameworks")

        # Summarize
        result = await client.summarize("https://example.com/article")

        # Enrich (news)
        result = await client.enrich("AI agents")
    """

    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    summarizer_engine: str | None = None
    _resolved_key: str | None = field(init=False, repr=False, default=None)
    _resolved_engine: SummarizerEngine = field(init=False, repr=False, default=SummarizerEngine.CECIL)

    def __post_init__(self) -> None:
        self._resolved_key = _get_api_key(self.api_key)
        engine_str = self.summarizer_engine or os.environ.get("KAGI_SUMMARIZER_ENGINE", "cecil")
        try:
            self._resolved_engine = SummarizerEngine(engine_str)
        except ValueError:
            self._resolved_engine = SummarizerEngine.CECIL

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {"Authorization": f"Bot {self._resolved_key}"}

    def _check_key(self) -> Result[None, KagiError]:
        """Verify API key is available."""
        if not self._resolved_key:
            return Err(KagiError(ErrorCode.CONFIG_ERROR, "KAGI_API_KEY not configured"))
        return Ok(None)

    # ------------------------------------------------------------------
    # Search API (requires beta access)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> Result[SearchResponse, KagiError]:
        """Search the web using Kagi Search API.

        Note: Search API is in closed beta. Email support@kagi.com for access.
        Consider using fastgpt() as an alternative.
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

                err = _check_status(resp)
                if err:
                    return Err(err)

                data = resp.json()
                results = [
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        published=item.get("published"),
                    )
                    for item in data.get("data", [])
                    if item.get("t") == 0  # t=0: search result, t=1: related searches
                ]

                return Ok(SearchResponse(
                    query=query,
                    results=results[:limit],
                    result_count=len(results),
                ))

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))

    # ------------------------------------------------------------------
    # FastGPT API (AI answer with references — no beta required)
    # ------------------------------------------------------------------

    async def fastgpt(
        self,
        query: str,
        cache: bool = True,
    ) -> Result[FastGPTResponse, KagiError]:
        """Get an AI-generated answer with references using Kagi FastGPT.

        Unlike search(), this endpoint is generally available and does not
        require Search API beta access. Returns a synthesized answer with
        cited references.

        Args:
            query: Question or search query
            cache: Whether to use cached results (default: True)
        """
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        payload: dict[str, Any] = {"query": query}
        if not cache:
            payload["cache"] = "false"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{KAGI_API_BASE}/fastgpt",
                    json=payload,
                    headers=self._headers(),
                )

                err = _check_status(resp)
                if err:
                    return Err(err)

                data = resp.json()
                gpt_data = data.get("data", {})

                references = [
                    FastGPTReference(
                        title=ref.get("title", ""),
                        snippet=ref.get("snippet", ""),
                        url=ref.get("url", ""),
                    )
                    for ref in gpt_data.get("references", [])
                ]

                return Ok(FastGPTResponse(
                    query=query,
                    output=gpt_data.get("output", ""),
                    references=references,
                    tokens=gpt_data.get("tokens", 0),
                ))

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))

    # ------------------------------------------------------------------
    # Summarizer API
    # ------------------------------------------------------------------

    async def summarize(
        self,
        url: str = "",
        text: str = "",
        summary_type: Literal["summary", "takeaway"] | SummaryType = SummaryType.SUMMARY,
        engine: Literal["cecil", "agnes", "daphne", "muriel"] | SummarizerEngine | None = None,
        target_language: str | None = None,
        cache: bool | None = None,
    ) -> Result[SummaryResponse, KagiError]:
        """Summarize content from a URL or raw text.

        Args:
            url: URL to summarize (mutually exclusive with text)
            text: Raw text to summarize (mutually exclusive with url)
            summary_type: "summary" for prose, "takeaway" for bullet points
            engine: Summarizer engine (defaults to KAGI_SUMMARIZER_ENGINE env or "cecil")
            target_language: Optional language code (e.g., "EN", "DE")
            cache: Whether to use cached results (None = server default)
        """
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        if url and text:
            return Err(KagiError(ErrorCode.USER_ERROR, "url and text are mutually exclusive"))
        if not url and not text:
            return Err(KagiError(ErrorCode.USER_ERROR, "Either url or text is required"))

        # Normalize enums
        if isinstance(summary_type, str):
            summary_type = SummaryType(summary_type)
        if engine is None:
            engine = self._resolved_engine
        elif isinstance(engine, str):
            engine = SummarizerEngine(engine)

        # Build params — kagiapi uses GET with query params, not POST
        params: dict[str, str] = {
            "engine": engine.value,
            "summary_type": summary_type.value,
        }
        if url:
            params["url"] = url
        else:
            params["text"] = text
        if target_language:
            params["target_language"] = target_language
        if cache is not None:
            params["cache"] = "true" if cache else "false"

        try:
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                resp = await client.get(
                    f"{KAGI_API_BASE}/summarize",
                    params=params,
                    headers=self._headers(),
                )

                err = _check_status(resp)
                if err:
                    return Err(err)

                data = resp.json()
                output = data.get("data", {}).get("output", "")

                return Ok(SummaryResponse(
                    url=url or "(text input)",
                    summary=output,
                    summary_type=summary_type,
                    engine=engine,
                ))

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))

    # ------------------------------------------------------------------
    # Enrich API (news)
    # ------------------------------------------------------------------

    async def enrich(
        self,
        query: str,
    ) -> Result[EnrichResponse, KagiError]:
        """Search news using Kagi Enrich API.

        Args:
            query: News search query
        """
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{KAGI_API_BASE}/enrich/news",
                    params={"q": query},
                    headers=self._headers(),
                )

                err = _check_status(resp)
                if err:
                    return Err(err)

                data = resp.json()
                results = [
                    EnrichResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        published=item.get("published"),
                    )
                    for item in data.get("data", [])
                    if item.get("t") == 0
                ]

                return Ok(EnrichResponse(
                    query=query,
                    results=results,
                    result_count=len(results),
                ))

        except httpx.TimeoutException:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, "Request timed out"))
        except httpx.RequestError as e:
            return Err(KagiError(ErrorCode.BACKEND_ERROR, f"Request failed: {e}"))


# Convenience functions
async def search(query: str, limit: int = 10, api_key: str | None = None) -> Result[SearchResponse, KagiError]:
    """Search the web using Kagi Search API."""
    return await KagiClient(api_key=api_key).search(query, limit)


async def fastgpt(query: str, api_key: str | None = None) -> Result[FastGPTResponse, KagiError]:
    """Get an AI answer with references using Kagi FastGPT."""
    return await KagiClient(api_key=api_key).fastgpt(query)


async def summarize(
    url: str,
    summary_type: Literal["summary", "takeaway"] = "summary",
    engine: Literal["cecil", "agnes", "daphne", "muriel"] | None = None,
    target_language: str | None = None,
    api_key: str | None = None,
) -> Result[SummaryResponse, KagiError]:
    """Summarize content from a URL."""
    return await KagiClient(api_key=api_key).summarize(url, summary_type=summary_type, engine=engine, target_language=target_language)


async def enrich(query: str, api_key: str | None = None) -> Result[EnrichResponse, KagiError]:
    """Search news using Kagi Enrich API."""
    return await KagiClient(api_key=api_key).enrich(query)

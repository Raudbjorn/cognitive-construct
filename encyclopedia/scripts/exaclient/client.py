"""Exa AI client implementation.

All search methods are async, using httpx.AsyncClient.
Sync convenience functions at module level use asyncio.run() for CLI usage.
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import Any

import httpx

from .config import API_CONFIG
from .result import Result, Ok, Err
from .types import (
    SearchType,
    LivecrawlMode,
    WebSearchOptions,
    CodeSearchOptions,
    WebSearchResult,
    CodeSearchResult,
    ExaSearchResponse,
    ExaSearchResultItem,
    ExaCodeResponse,
)

logger = logging.getLogger(__name__)


class ExaClient:
    """Async client for Exa AI search API.

    All search methods are coroutines. For sync CLI usage, use the
    module-level convenience functions (web_search, code_search).

    Usage:
        client = ExaClient(api_key="your-key")  # or set EXA_API_KEY env var

        # Async web search
        result = await client.web_search("latest python frameworks")
        if result.is_ok():
            print(result.value.context)
        else:
            print(f"Error: {result.error}")

        # Async code search
        result = await client.code_search("python async http client")
        if result.is_ok():
            print(result.value.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        integration_name: str = "exaclient-python",
    ) -> None:
        self._api_key = api_key or os.environ.get("EXA_API_KEY")
        self._base_url = base_url or API_CONFIG.BASE_URL
        self._integration_name = integration_name

    def _get_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ValueError("API key required: set EXA_API_KEY or pass api_key to constructor")
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "x-exa-integration": self._integration_name,
        }

    def _build_search_request(
        self,
        query: str,
        options: WebSearchOptions,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "type": options.search_type.value,
            "numResults": options.num_results,
            "contents": {
                "text": True,
                "context": {"maxCharacters": options.max_chars},
                "livecrawl": options.livecrawl.value,
            },
        }
        if options.include_domains:
            payload["includeDomains"] = options.include_domains
        if options.exclude_domains:
            payload["excludeDomains"] = options.exclude_domains
        if options.start_published_date:
            payload["startPublishedDate"] = options.start_published_date
        if options.end_published_date:
            payload["endPublishedDate"] = options.end_published_date
        if options.category:
            payload["category"] = options.category
        return payload

    def _parse_search_response(self, data: dict[str, Any]) -> ExaSearchResponse:
        results = [
            ExaSearchResultItem(
                id=r.get("id", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                text=r.get("text", ""),
                published_date=r.get("publishedDate", ""),
                author=r.get("author", ""),
                summary=r.get("summary"),
                image=r.get("image"),
                favicon=r.get("favicon"),
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]
        return ExaSearchResponse(
            request_id=data.get("requestId", ""),
            results=results,
            context=data.get("context"),
            autoprompt_string=data.get("autopromptString"),
            resolved_search_type=data.get("resolvedSearchType", ""),
        )

    def _parse_code_response(self, data: dict[str, Any]) -> ExaCodeResponse:
        return ExaCodeResponse(
            request_id=data.get("requestId", ""),
            query=data.get("query", ""),
            response=data.get("response", ""),
            results_count=data.get("resultsCount", 0),
            cost_dollars=data.get("costDollars", "0"),
            search_time=data.get("searchTime", 0.0),
            repository=data.get("repository"),
            output_tokens=data.get("outputTokens"),
            traces=data.get("traces"),
        )

    async def web_search(
        self,
        query: str,
        options: WebSearchOptions | None = None,
    ) -> Result[WebSearchResult, str]:
        """Perform a web search using Exa AI.

        Args:
            query: Search query string.
            options: Search options (defaults to WebSearchOptions()).

        Returns:
            Result containing WebSearchResult on success, error message on failure.
        """
        if not self._api_key:
            return Err("EXA_API_KEY environment variable or api_key is required")

        opts = options or WebSearchOptions()
        payload = self._build_search_request(query, opts)

        try:
            async with httpx.AsyncClient(timeout=API_CONFIG.WEB_SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}{API_CONFIG.SEARCH_ENDPOINT}",
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            parsed = self._parse_search_response(data)
            if not parsed.context:
                return Err("No search results found")

            return Ok(WebSearchResult(context=parsed.context, response=parsed))

        except httpx.HTTPStatusError as e:
            error_msg = str(e)
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", str(e))
            except Exception:
                pass
            return Err(f"Search error ({e.response.status_code}): {error_msg}")
        except httpx.TimeoutException:
            return Err("Search request timed out")
        except Exception as e:
            logger.exception("Unexpected error during web search")
            return Err(f"Search error: {e}")

    async def code_search(
        self,
        query: str,
        options: CodeSearchOptions | None = None,
    ) -> Result[CodeSearchResult, str]:
        """Search for code examples and documentation.

        Args:
            query: Search query for code/docs.
            options: Code search options (defaults to CodeSearchOptions()).

        Returns:
            Result containing CodeSearchResult on success, error message on failure.
        """
        if not self._api_key:
            return Err("EXA_API_KEY environment variable or api_key is required")

        opts = options or CodeSearchOptions()
        payload: dict[str, Any] = {
            "query": query,
            "tokensNum": opts.tokens,
        }
        if opts.flags:
            payload["flags"] = opts.flags

        try:
            async with httpx.AsyncClient(timeout=API_CONFIG.CODE_SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}{API_CONFIG.CONTEXT_ENDPOINT}",
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            parsed = self._parse_code_response(data)
            content = parsed.response
            if not content:
                return Err("No code context found")

            return Ok(CodeSearchResult(content=content, response=parsed))

        except httpx.HTTPStatusError as e:
            error_msg = str(e)
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", str(e))
            except Exception:
                pass
            return Err(f"Code search error ({e.response.status_code}): {error_msg}")
        except httpx.TimeoutException:
            return Err("Code search request timed out")
        except Exception as e:
            logger.exception("Unexpected error during code search")
            return Err(f"Code search error: {e}")



# ---------------------------------------------------------------------------
# Sync convenience functions for CLI / standalone usage
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Result[WebSearchResult, str]:
    """Perform a web search (sync wrapper for CLI usage)."""
    client = ExaClient(api_key=api_key)
    options = WebSearchOptions(**kwargs) if kwargs else None
    return asyncio.run(client.web_search(query, options))


def code_search(
    query: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Result[CodeSearchResult, str]:
    """Search for code examples (sync wrapper for CLI usage)."""
    client = ExaClient(api_key=api_key)
    options = CodeSearchOptions(**kwargs) if kwargs else None
    return asyncio.run(client.code_search(query, options))

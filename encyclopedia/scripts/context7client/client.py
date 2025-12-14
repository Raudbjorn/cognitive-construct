"""Context7 API client library."""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .result import Err, Ok, Result
from .types import (
    APIResponseMetadata,
    CodeDocsResponse,
    CodeExample,
    CodeSnippet,
    Context7Error,
    DocsFormat,
    DocsMode,
    GetDocsOptions,
    InfoDocsResponse,
    InfoSnippet,
    LibraryState,
    Pagination,
    SearchLibraryResponse,
    SearchResult,
    TextDocsResponse,
)

DEFAULT_BASE_URL = "https://context7.com/api"
API_KEY_PREFIX = "ctx7sk"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 5


def _get_api_key(api_key: str | None = None) -> str | None:
    """Get API key from parameter or environment."""
    return api_key or os.environ.get("CONTEXT7_API_KEY")


def _backoff(retry_count: int) -> float:
    """Calculate exponential backoff delay in seconds."""
    return math.exp(retry_count) * 0.05  # 50ms base


def _parse_search_result(data: dict[str, Any]) -> SearchResult:
    """Parse a search result from API response."""
    return SearchResult(
        id=data.get("id", ""),
        title=data.get("title", ""),
        description=data.get("description", ""),
        branch=data.get("branch", ""),
        last_update_date=data.get("lastUpdateDate", ""),
        state=LibraryState(data.get("state", "initial")),
        total_tokens=data.get("totalTokens", 0),
        total_snippets=data.get("totalSnippets", 0),
        stars=data.get("stars"),
        trust_score=data.get("trustScore"),
        benchmark_score=data.get("benchmarkScore"),
        versions=data.get("versions", []),
    )


def _parse_pagination(data: dict[str, Any]) -> Pagination:
    """Parse pagination from API response."""
    return Pagination(
        page=data.get("page", 1),
        limit=data.get("limit", 10),
        total_pages=data.get("totalPages", 1),
        has_next=data.get("hasNext", False),
        has_prev=data.get("hasPrev", False),
    )


def _parse_code_snippet(data: dict[str, Any]) -> CodeSnippet:
    """Parse a code snippet from API response."""
    code_list = [
        CodeExample(language=c.get("language", ""), code=c.get("code", ""))
        for c in data.get("codeList", [])
    ]
    return CodeSnippet(
        code_title=data.get("codeTitle", ""),
        code_description=data.get("codeDescription", ""),
        code_language=data.get("codeLanguage", ""),
        code_tokens=data.get("codeTokens", 0),
        code_id=data.get("codeId", ""),
        page_title=data.get("pageTitle", ""),
        code_list=code_list,
    )


def _parse_info_snippet(data: dict[str, Any]) -> InfoSnippet:
    """Parse an info snippet from API response."""
    return InfoSnippet(
        content=data.get("content", ""),
        content_tokens=data.get("contentTokens", 0),
        page_id=data.get("pageId"),
        breadcrumb=data.get("breadcrumb"),
    )


@dataclass
class Context7Client:
    """Context7 API client for library documentation lookup.

    Usage:
        client = Context7Client(api_key="ctx7sk...")  # or set CONTEXT7_API_KEY env var

        # Search for a library
        result = await client.search_library("react")
        if result.is_ok():
            for lib in result.value.results:
                print(f"{lib.title}: {lib.id}")

        # Get documentation
        result = await client.get_docs("/facebook/react", topic="hooks")
        if result.is_ok():
            for snippet in result.value.snippets:
                print(f"{snippet.code_title}")
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._resolved_key = _get_api_key(self.api_key)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self._resolved_key}",
            "Content-Type": "application/json",
        }

    def _check_key(self) -> Result[None, Context7Error]:
        """Verify API key is available."""
        if not self._resolved_key:
            return Err(
                Context7Error(
                    "API key is required. Pass api_key or set CONTEXT7_API_KEY environment variable."
                )
            )
        if not self._resolved_key.startswith(API_KEY_PREFIX):
            # Warning only, don't fail
            pass
        return Ok(None)

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Result[dict[str, Any] | str, Context7Error]:
        """Make HTTP request with retry logic."""
        key_check = self._check_key()
        if key_check.is_err():
            return key_check

        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        # Build query string for GET requests
        params = None
        if method == "GET" and query:
            params = {k: v for k, v in query.items() if v is not None}

        last_error: Context7Error | None = None

        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        response = await client.get(
                            url, params=params, headers=self._headers()
                        )
                    else:
                        response = await client.post(
                            url, json=body, headers=self._headers()
                        )

                    if not response.is_success:
                        try:
                            error_body = response.json()
                            error_msg = error_body.get("error") or error_body.get("message") or response.reason_phrase
                        except json.JSONDecodeError:
                            error_msg = response.reason_phrase
                        return Err(Context7Error(error_msg, response.status_code))

                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        return Ok(response.json())
                    else:
                        # Text response - extract pagination headers
                        return Ok({
                            "_text": response.text,
                            "_headers": {
                                "page": response.headers.get("x-context7-page"),
                                "limit": response.headers.get("x-context7-limit"),
                                "totalPages": response.headers.get("x-context7-total-pages"),
                                "hasNext": response.headers.get("x-context7-has-next"),
                                "hasPrev": response.headers.get("x-context7-has-prev"),
                                "totalTokens": response.headers.get("x-context7-total-tokens"),
                            },
                        })

            except httpx.TimeoutException:
                last_error = Context7Error("Request timed out")
            except httpx.RequestError as e:
                last_error = Context7Error(f"Request failed: {e}")

            if attempt < self.retries:
                await asyncio.sleep(_backoff(attempt))

        return Err(last_error or Context7Error("Exhausted all retries"))

    async def search_library(
        self, query: str
    ) -> Result[SearchLibraryResponse, Context7Error]:
        """Search for libraries by name.

        Args:
            query: Library name to search for (e.g., "react", "svelte", "pandas")

        Returns:
            Result containing SearchLibraryResponse on success
        """
        result = await self._request("POST", "v1/search", body={"query": query})
        if result.is_err():
            return result

        data = result.value
        if isinstance(data, str):
            return Err(Context7Error("Unexpected text response"))

        results = [_parse_search_result(r) for r in data.get("results", [])]
        metadata = APIResponseMetadata(
            authentication=data.get("metadata", {}).get("authentication", "none")
        )

        return Ok(SearchLibraryResponse(results=results, metadata=metadata))

    async def get_docs(
        self,
        library_id: str,
        topic: str | None = None,
        mode: DocsMode | str = DocsMode.CODE,
        format: DocsFormat | str = DocsFormat.JSON,
        version: str | None = None,
        page: int | None = None,
        limit: int | None = None,
    ) -> Result[CodeDocsResponse | InfoDocsResponse | TextDocsResponse, Context7Error]:
        """Get documentation for a library.

        Args:
            library_id: Context7 library ID (e.g., "/facebook/react")
            topic: Topic to focus on (e.g., "hooks", "routing")
            mode: "code" for API refs, "info" for conceptual guides
            format: "json" for structured, "txt" for plain text
            version: Specific library version
            page: Page number for pagination
            limit: Results per page

        Returns:
            Result containing docs response based on mode/format
        """
        # Normalize enums
        mode_str = mode.value if isinstance(mode, DocsMode) else mode
        format_str = format.value if isinstance(format, DocsFormat) else format

        query = {
            "topic": topic,
            "mode": mode_str,
            "format": format_str,
            "version": version,
            "page": page,
            "limit": limit,
        }

        # Library ID becomes the path
        path = f"v1{library_id}" if library_id.startswith("/") else f"v1/{library_id}"

        result = await self._request("GET", path, query=query)
        if result.is_err():
            return result

        data = result.value

        # Handle text response
        if isinstance(data, dict) and "_text" in data:
            headers = data.get("_headers", {})
            pagination = Pagination(
                page=int(headers.get("page") or 1),
                limit=int(headers.get("limit") or 10),
                total_pages=int(headers.get("totalPages") or 1),
                has_next=headers.get("hasNext") == "true",
                has_prev=headers.get("hasPrev") == "true",
            )
            return Ok(
                TextDocsResponse(
                    content=data["_text"],
                    pagination=pagination,
                    total_tokens=int(headers.get("totalTokens") or 0),
                )
            )

        # Handle JSON response
        pagination = _parse_pagination(data.get("pagination", {}))
        total_tokens = data.get("totalTokens", 0)

        if mode_str == "info":
            snippets = [_parse_info_snippet(s) for s in data.get("snippets", [])]
            return Ok(
                InfoDocsResponse(
                    snippets=snippets, pagination=pagination, total_tokens=total_tokens
                )
            )
        else:
            snippets = [_parse_code_snippet(s) for s in data.get("snippets", [])]
            return Ok(
                CodeDocsResponse(
                    snippets=snippets, pagination=pagination, total_tokens=total_tokens
                )
            )

    async def resolve_library_id(
        self, library_name: str
    ) -> Result[str | None, Context7Error]:
        """Resolve a library name to its Context7 ID.

        Convenience method that searches and returns the best match ID.

        Args:
            library_name: Library name (e.g., "react", "svelte")

        Returns:
            Result containing library ID (e.g., "/facebook/react") or None if not found
        """
        result = await self.search_library(library_name)
        if result.is_err():
            return result

        if result.value.results:
            return Ok(result.value.results[0].id)
        return Ok(None)


# Convenience functions
async def search_library(
    query: str, api_key: str | None = None
) -> Result[SearchLibraryResponse, Context7Error]:
    """Search for libraries by name."""
    client = Context7Client(api_key=api_key)
    return await client.search_library(query)


async def get_docs(
    library_id: str,
    topic: str | None = None,
    mode: str = "code",
    api_key: str | None = None,
) -> Result[CodeDocsResponse | InfoDocsResponse | TextDocsResponse, Context7Error]:
    """Get documentation for a library."""
    client = Context7Client(api_key=api_key)
    return await client.get_docs(library_id, topic=topic, mode=DocsMode(mode))

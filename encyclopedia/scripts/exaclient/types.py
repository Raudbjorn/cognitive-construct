"""Type definitions for Exa AI API."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SearchType(str, Enum):
    """Search type options."""
    AUTO = "auto"
    FAST = "fast"
    DEEP = "deep"


class LivecrawlMode(str, Enum):
    """Live crawl mode options."""
    FALLBACK = "fallback"
    PREFERRED = "preferred"
    ALWAYS = "always"


# Request options
@dataclass(frozen=True, slots=True)
class WebSearchOptions:
    """Options for web search."""
    num_results: int = 8
    search_type: SearchType = SearchType.AUTO
    livecrawl: LivecrawlMode = LivecrawlMode.FALLBACK
    max_chars: int = 10000
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    start_published_date: str | None = None
    end_published_date: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class CodeSearchOptions:
    """Options for code search."""
    tokens: int = 5000
    flags: list[str] | None = None


# API Response types
@dataclass(slots=True)
class ExaSearchResultItem:
    """Individual search result."""
    id: str
    title: str
    url: str
    text: str
    published_date: str = ""
    author: str = ""
    summary: str | None = None
    image: str | None = None
    favicon: str | None = None
    score: float | None = None


@dataclass(slots=True)
class ExaSearchResponse:
    """Response from Exa search API."""
    request_id: str
    results: list[ExaSearchResultItem]
    context: str | None = None
    autoprompt_string: str | None = None
    resolved_search_type: str = ""


@dataclass(slots=True)
class ExaCodeResponse:
    """Response from Exa code/context API."""
    request_id: str
    query: str
    response: str
    results_count: int
    cost_dollars: str
    search_time: float
    repository: str | None = None
    output_tokens: int | None = None
    traces: Any | None = None


# Client result types
@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """Result of a web search operation."""
    context: str
    response: ExaSearchResponse


@dataclass(frozen=True, slots=True)
class CodeSearchResult:
    """Result of a code search operation."""
    content: str
    response: ExaCodeResponse

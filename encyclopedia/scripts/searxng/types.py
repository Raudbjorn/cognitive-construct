"""Type definitions for SearXNG client."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Individual search result."""

    url: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class InfoboxUrl:
    """URL within an infobox."""

    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Infobox:
    """Knowledge panel / infobox from search."""

    infobox: str
    id: str
    content: str
    urls: list[InfoboxUrl] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Full search response."""

    query: str
    number_of_results: int
    results: list[SearchResult]
    infoboxes: list[Infobox] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearxngError:
    """Error from SearXNG operations."""

    message: str
    status_code: int | None = None

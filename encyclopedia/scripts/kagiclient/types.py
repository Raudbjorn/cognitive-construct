"""Type definitions for Kagi API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SummaryType(str, Enum):
    """Summary output format."""

    SUMMARY = "summary"  # Prose paragraph
    TAKEAWAY = "takeaway"  # Bullet points


class SummarizerEngine(str, Enum):
    """Available summarizer engines."""

    CECIL = "cecil"  # Friendly, descriptive (default)
    AGNES = "agnes"  # Formal, technical
    DAPHNE = "daphne"  # Creative, informal
    MURIEL = "muriel"  # Best quality, slowest


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Individual search result."""

    title: str
    url: str
    snippet: str
    published: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Search API response."""

    query: str
    results: list[SearchResult]
    result_count: int


@dataclass(frozen=True, slots=True)
class SummaryResponse:
    """Summarize API response."""

    url: str
    summary: str
    summary_type: SummaryType
    engine: SummarizerEngine


@dataclass(frozen=True, slots=True)
class KagiError:
    """API error details."""

    code: int
    message: str


class ErrorCode:
    """Error code constants."""

    USER_ERROR = 1
    CONFIG_ERROR = 2
    BACKEND_ERROR = 3
    INTERNAL_ERROR = 4

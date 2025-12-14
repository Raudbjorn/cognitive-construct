"""Type definitions for Context7 API client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class LibraryState(str, Enum):
    """State of a library in Context7."""

    INITIAL = "initial"
    FINALIZED = "finalized"
    PROCESSING = "processing"
    ERROR = "error"
    DELETE = "delete"


class DocsMode(str, Enum):
    """Documentation mode."""

    CODE = "code"  # API references and code examples
    INFO = "info"  # Conceptual guides and narrative


class DocsFormat(str, Enum):
    """Response format."""

    JSON = "json"
    TXT = "txt"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A library search result."""

    id: str
    title: str
    description: str
    branch: str
    last_update_date: str
    state: LibraryState
    total_tokens: int
    total_snippets: int
    stars: int | None = None
    trust_score: float | None = None
    benchmark_score: float | None = None
    versions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class APIResponseMetadata:
    """API response metadata."""

    authentication: Literal["none", "personal", "team"]


@dataclass(frozen=True, slots=True)
class SearchLibraryResponse:
    """Search library response."""

    results: list[SearchResult]
    metadata: APIResponseMetadata


@dataclass(frozen=True, slots=True)
class CodeExample:
    """A code example within a snippet."""

    language: str
    code: str


@dataclass(frozen=True, slots=True)
class CodeSnippet:
    """A code documentation snippet."""

    code_title: str
    code_description: str
    code_language: str
    code_tokens: int
    code_id: str
    page_title: str
    code_list: list[CodeExample] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InfoSnippet:
    """An informational documentation snippet."""

    content: str
    content_tokens: int
    page_id: str | None = None
    breadcrumb: str | None = None


@dataclass(frozen=True, slots=True)
class Pagination:
    """Pagination information."""

    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool


@dataclass(frozen=True, slots=True)
class CodeDocsResponse:
    """Response containing code documentation snippets."""

    snippets: list[CodeSnippet]
    pagination: Pagination
    total_tokens: int


@dataclass(frozen=True, slots=True)
class InfoDocsResponse:
    """Response containing informational documentation snippets."""

    snippets: list[InfoSnippet]
    pagination: Pagination
    total_tokens: int


@dataclass(frozen=True, slots=True)
class TextDocsResponse:
    """Response containing plain text documentation."""

    content: str
    pagination: Pagination
    total_tokens: int


@dataclass(frozen=True, slots=True)
class GetDocsOptions:
    """Options for getting documentation."""

    version: str | None = None
    page: int | None = None
    topic: str | None = None
    limit: int | None = None
    mode: DocsMode = DocsMode.CODE
    format: DocsFormat = DocsFormat.JSON


@dataclass(frozen=True, slots=True)
class Context7Error:
    """API error details."""

    message: str
    status_code: int | None = None

"""Type definitions for the Claude Skills MCP backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# === Enums ===


class SourceType(str, Enum):
    """Skill source type."""

    GITHUB = "github"
    LOCAL = "local"


class DocumentType(str, Enum):
    """Document type."""

    TEXT = "text"
    IMAGE = "image"


# === Error Types ===


@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadError:
    """Skill loading error."""

    message: str
    source: str
    cause: str | None = None


@dataclass(frozen=True, slots=True)
class SearchError:
    """Search operation error."""

    message: str
    query: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigError:
    """Configuration error."""

    message: str
    path: str | None = None


# === Configuration Types ===


@dataclass(frozen=True, slots=True)
class SkillSourceConfig:
    """Configuration for a skill source."""

    type: SourceType
    url: str | None = None
    path: str | None = None
    subpath: str = ""


@dataclass(frozen=True, slots=True)
class Config:
    """Application configuration."""

    skill_sources: tuple[SkillSourceConfig, ...]
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 3
    max_skill_content_chars: int | None = None
    load_skill_documents: bool = True
    max_image_size_bytes: int = 5242880  # 5MB
    allowed_image_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
    )
    text_file_extensions: tuple[str, ...] = (
        ".md",
        ".py",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".sh",
        ".r",
        ".ipynb",
        ".xml",
    )
    auto_update_enabled: bool = True
    auto_update_interval_minutes: int = 60
    github_api_token: str | None = None


# === Document Types ===


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata for a skill document (before content is fetched)."""

    type: DocumentType
    size: int
    url: str | None = None
    fetched: bool = False


@dataclass(frozen=True, slots=True)
class DocumentContent:
    """Full document content."""

    type: DocumentType
    content: str | None = None
    size: int = 0
    url: str | None = None
    size_exceeded: bool = False
    fetched: bool = True


# === Skill Types ===


@dataclass
class Skill:
    """Represents a Claude Agent Skill.

    This is a mutable dataclass because it maintains document caching state.

    Attributes
    ----------
    name : str
        Skill name.
    description : str
        Short description of the skill.
    content : str
        Full content of the SKILL.md file.
    source : str
        Origin of the skill (GitHub URL or local path).
    documents : dict[str, dict[str, Any]]
        Additional documents from the skill directory.
        Keys are relative paths, values contain metadata and content.
    """

    name: str
    description: str
    content: str
    source: str
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    _document_fetcher: Callable[[str], dict[str, Any] | None] | None = field(
        default=None, repr=False
    )
    _document_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )

    def get_document(self, doc_path: str) -> dict[str, Any] | None:
        """Fetch document content on-demand with caching.

        Parameters
        ----------
        doc_path : str
            Relative path to the document.

        Returns
        -------
        dict[str, Any] | None
            Document content with metadata, or None if not found.
        """
        # Check memory cache first
        if doc_path in self._document_cache:
            return self._document_cache[doc_path]

        # Check if document exists in metadata
        if doc_path not in self.documents:
            return None

        # If already fetched (eager loaded), return from documents
        doc_info = self.documents[doc_path]
        if doc_info.get("fetched") or "content" in doc_info:
            return doc_info

        # Fetch using the document_fetcher (lazy loading)
        if self._document_fetcher:
            content = self._document_fetcher(doc_path)
            if content:
                # Cache it in memory
                self._document_cache[doc_path] = content
                return content

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert skill to dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary with skill information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "source": self.source,
            "documents": self.documents,
        }


# === Search Types ===


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Individual search result."""

    name: str
    description: str
    content: str
    source: str
    relevance_score: float
    documents: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Search operation response."""

    query: str
    results: tuple[SearchResult, ...]
    total_count: int


# === Loading State ===


@dataclass
class LoadingState:
    """Thread-safe state tracker for background skill loading.

    This is a mutable dataclass because it tracks loading progress.

    Attributes
    ----------
    total_skills : int
        Total number of skills expected to be loaded.
    loaded_skills : int
        Number of skills loaded so far.
    is_complete : bool
        Whether loading is complete.
    errors : list[str]
        List of error messages encountered during loading.
    """

    total_skills: int = 0
    loaded_skills: int = 0
    is_complete: bool = False
    errors: list[str] = field(default_factory=list)
    _lock: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the threading lock."""
        import threading

        self._lock = threading.Lock()

    def update_progress(self, loaded: int, total: int | None = None) -> None:
        """Update loading progress.

        Parameters
        ----------
        loaded : int
            Number of skills loaded.
        total : int | None, optional
            Total skills expected (if known), by default None.
        """
        with self._lock:
            self.loaded_skills = loaded
            if total is not None:
                self.total_skills = total

    def add_error(self, error: str) -> None:
        """Add an error message.

        Parameters
        ----------
        error : str
            Error message to record.
        """
        with self._lock:
            self.errors.append(error)

    def mark_complete(self) -> None:
        """Mark loading as complete."""
        with self._lock:
            self.is_complete = True

    def get_status_message(self) -> str | None:
        """Get current loading status message.

        Returns
        -------
        str | None
            Status message if loading is in progress, None if complete.
        """
        with self._lock:
            if self.is_complete:
                return None

            if self.loaded_skills == 0:
                return "[LOADING: Skills are being loaded in the background, please wait...]\n"

            if self.total_skills > 0:
                return (
                    f"[LOADING: {self.loaded_skills}/{self.total_skills} skills loaded, "
                    "indexing in progress...]\n"
                )
            return (
                f"[LOADING: {self.loaded_skills} skills loaded so far, "
                "indexing in progress...]\n"
            )

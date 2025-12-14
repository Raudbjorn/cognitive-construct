"""Type definitions for the Mem0 API client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Enums ===


class MemoryEvent(str, Enum):
    """Memory operation event types."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class EntityType(str, Enum):
    """Entity types in the system."""

    USER = "user"
    AGENT = "agent"
    APP = "app"
    RUN = "run"


class FeedbackValue(str, Enum):
    """Feedback values for memories."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    VERY_NEGATIVE = "VERY_NEGATIVE"


class MemberRole(str, Enum):
    """Project member roles."""

    READER = "READER"
    OWNER = "OWNER"


# === Error Type ===


@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


# === Memory Types ===


@dataclass(frozen=True, slots=True)
class Memory:
    """A single memory entry."""

    id: str
    memory: str
    hash: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    categories: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MemoryResult:
    """Result of a memory operation (add/update/delete)."""

    id: str
    event: MemoryEvent
    memory: str


@dataclass(frozen=True, slots=True)
class MemoryAddResponse:
    """Response from adding memories."""

    results: list[MemoryResult]


@dataclass(frozen=True, slots=True)
class MemoryListResponse:
    """Response containing a list of memories."""

    results: list[Memory]
    page: int | None = None
    page_size: int | None = None
    total: int | None = None


@dataclass(frozen=True, slots=True)
class MemorySearchResponse:
    """Response from memory search."""

    results: list[Memory]


@dataclass(frozen=True, slots=True)
class MemoryHistoryEntry:
    """A single entry in memory history."""

    id: str
    memory_id: str
    old_memory: str | None
    new_memory: str | None
    event: str
    created_at: str
    is_deleted: bool = False


@dataclass(frozen=True, slots=True)
class MemoryDeleteResponse:
    """Response from delete operation."""

    message: str


@dataclass(frozen=True, slots=True)
class MemoryUpdateResponse:
    """Response from update operation."""

    id: str
    memory: str
    updated_at: str | None = None


# === Entity Types ===


@dataclass(frozen=True, slots=True)
class Entity:
    """An entity (user, agent, app, run)."""

    name: str
    type: EntityType
    memory_count: int | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class EntitiesResponse:
    """Response containing entities."""

    results: list[Entity]


# === Batch Operation Types ===


@dataclass(frozen=True, slots=True)
class BatchMemoryUpdate:
    """Single memory update in batch operation."""

    memory_id: str
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BatchMemoryDelete:
    """Single memory delete in batch operation."""

    memory_id: str


@dataclass(frozen=True, slots=True)
class BatchUpdateResponse:
    """Response from batch update."""

    updated: int
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BatchDeleteResponse:
    """Response from batch delete."""

    deleted: int
    errors: list[dict[str, Any]] = field(default_factory=list)


# === Export Types ===


@dataclass(frozen=True, slots=True)
class ExportCreateResponse:
    """Response from creating an export."""

    request_id: str
    message: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ExportData:
    """Exported memory data."""

    data: dict[str, Any]
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ExportSummary:
    """Summary of exported data."""

    status: str
    summary: dict[str, Any] | None = None


# === Webhook Types ===


@dataclass(frozen=True, slots=True)
class Webhook:
    """Webhook configuration."""

    id: int
    url: str
    name: str
    event_types: list[str]
    project_id: str | None = None
    is_active: bool = True
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class WebhookListResponse:
    """Response containing webhooks."""

    results: list[Webhook]


@dataclass(frozen=True, slots=True)
class WebhookDeleteResponse:
    """Response from webhook deletion."""

    message: str


# === Project Types ===


@dataclass(frozen=True, slots=True)
class ProjectMember:
    """Project member information."""

    email: str
    role: MemberRole


@dataclass(frozen=True, slots=True)
class ProjectMembersResponse:
    """Response containing project members."""

    results: list[ProjectMember]


@dataclass(frozen=True, slots=True)
class Project:
    """Project configuration."""

    id: str
    name: str
    org_id: str
    custom_instructions: str | None = None
    custom_categories: list[str] | None = None
    retrieval_criteria: list[dict[str, Any]] | None = None
    enable_graph: bool = False
    version: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectCreateResponse:
    """Response from project creation."""

    id: str
    name: str
    message: str | None = None


# === Feedback Types ===


@dataclass(frozen=True, slots=True)
class FeedbackResponse:
    """Response from feedback submission."""

    message: str


# === Validation Types ===


@dataclass(frozen=True, slots=True)
class PingResponse:
    """Response from API ping/validation."""

    org_id: str | None = None
    project_id: str | None = None
    user_email: str | None = None

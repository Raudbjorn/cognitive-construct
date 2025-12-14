"""Type definitions for the OpenAI Web Search client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Enums ===

class ReasoningEffort(str, Enum):
    """Reasoning effort level for reasoning models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MINIMAL = "minimal"


class SearchType(str, Enum):
    """Web search API version."""

    PREVIEW = "web_search_preview"
    PREVIEW_2025_03_11 = "web_search_preview_2025_03_11"


class SearchContextSize(str, Enum):
    """Amount of context to include in search results."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# === Request Types ===

@dataclass(frozen=True, slots=True)
class UserLocation:
    """User location for localized search results."""

    city: str
    timezone: str
    country: str | None = None
    region: str | None = None
    type: str = "approximate"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API request."""
        d: dict[str, Any] = {
            "type": self.type,
            "city": self.city,
            "timezone": self.timezone,
        }
        if self.country:
            d["country"] = self.country
        if self.region:
            d["region"] = self.region
        return d


# === Response Types ===

@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Search result from OpenAI."""

    content: str


# === Error Types ===

@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

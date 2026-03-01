"""Type definitions for sequential thinking client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ThoughtInput:
    """Input for processing a thought step."""

    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: bool | None = None
    revises_thought: int | None = None
    branch_from_thought: int | None = None
    branch_id: str | None = None
    needs_more_thoughts: bool | None = None


@dataclass(frozen=True, slots=True)
class ThoughtResponse:
    """Response from processing a thought."""

    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    branches: list[str] = field(default_factory=list)
    thought_history_length: int = 0


@dataclass(frozen=True, slots=True)
class ThinkingError:
    """Error details for thinking operations."""

    message: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThoughtData:
    """Internal representation of a thought in history."""

    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: bool | None = None
    revises_thought: int | None = None
    branch_from_thought: int | None = None
    branch_id: str | None = None
    needs_more_thoughts: bool | None = None

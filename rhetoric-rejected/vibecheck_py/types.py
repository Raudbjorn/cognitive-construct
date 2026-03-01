"""Type definitions for vibecheck client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# === Enums ===


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


class LearningType(str, Enum):
    """Types of learning entries."""

    MISTAKE = "mistake"
    PREFERENCE = "preference"
    SUCCESS = "success"


class LearningCategory(str, Enum):
    """Standard learning categories."""

    COMPLEX_SOLUTION_BIAS = "Complex Solution Bias"
    FEATURE_CREEP = "Feature Creep"
    PREMATURE_IMPLEMENTATION = "Premature Implementation"
    MISALIGNMENT = "Misalignment"
    OVERTOOLING = "Overtooling"
    PREFERENCE = "Preference"
    SUCCESS = "Success"
    OTHER = "Other"


# === Request Types ===


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """Override for LLM model selection."""

    provider: LLMProvider | str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class VibeCheckInput:
    """Input for vibe check operation."""

    goal: str
    plan: str
    model_override: ModelOverride | None = None
    user_prompt: str | None = None
    progress: str | None = None
    uncertainties: list[str] = field(default_factory=list)
    task_context: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class VibeLearnInput:
    """Input for vibe learn operation."""

    mistake: str
    category: LearningCategory | str
    solution: str | None = None
    type: LearningType = LearningType.MISTAKE
    session_id: str | None = None


# === Response Types ===


@dataclass(frozen=True, slots=True)
class VibeCheckResponse:
    """Response from vibe check operation."""

    questions: str


@dataclass(frozen=True, slots=True)
class LearningEntry:
    """A learning entry in the vibe log."""

    type: LearningType
    category: str
    mistake: str
    timestamp: int
    solution: str | None = None


@dataclass(frozen=True, slots=True)
class CategorySummary:
    """Summary of a learning category."""

    category: str
    count: int
    recent_example: LearningEntry


@dataclass(frozen=True, slots=True)
class VibeLearnResponse:
    """Response from vibe learn operation."""

    added: bool
    current_tally: int
    top_categories: list[CategorySummary] = field(default_factory=list)
    already_known: bool = False


@dataclass(frozen=True, slots=True)
class ConstitutionResponse:
    """Response from constitution operations."""

    rules: list[str] = field(default_factory=list)


# === Error Types ===


@dataclass(frozen=True, slots=True)
class VibeCheckError:
    """Error details for vibe check operations."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


# === Internal Types ===


@dataclass(frozen=True, slots=True)
class Interaction:
    """A recorded interaction in history."""

    input: VibeCheckInput
    output: str
    timestamp: int


@dataclass(frozen=True, slots=True)
class VibeLog:
    """The vibe log structure."""

    mistakes: dict[str, dict[str, Any]]
    last_updated: int

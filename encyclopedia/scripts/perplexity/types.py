"""Type definitions for Perplexity API client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Model(str, Enum):
    """Available Perplexity models."""

    SONAR_SMALL = "llama-3.1-sonar-small-128k-online"
    SONAR_LARGE = "llama-3.1-sonar-large-128k-online"
    SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"


DEFAULT_MODEL = Model.SONAR_SMALL

AVAILABLE_MODELS = [m.value for m in Model]


@dataclass(frozen=True, slots=True)
class Message:
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Chat completion request payload."""

    messages: list[Message]
    model: str = DEFAULT_MODEL.value
    max_tokens: int | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    stream: bool = False


@dataclass(frozen=True, slots=True)
class Choice:
    """Chat completion choice."""

    index: int
    message: Message
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Chat completion response."""

    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    citations: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Get the content of the first choice."""
        if self.choices:
            return self.choices[0].message.content
        return ""


@dataclass(frozen=True, slots=True)
class PerplexityError:
    """API error details."""

    message: str
    status_code: int | None = None

"""Type definitions for the OpenAI client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Enums ===

class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# === Request Types ===

@dataclass(frozen=True, slots=True)
class Message:
    """A single message in the chat history."""

    role: Role | str
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    """Request parameters for chat completion."""

    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 500


# === Response Types ===

@dataclass(frozen=True, slots=True)
class Choice:
    """A single choice in the completion response."""

    index: int
    message: Message
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionResponse:
    """Response from chat completion API."""

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, Any] | None = None


# === Error Types ===

@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

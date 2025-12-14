"""Type definitions for Cross LLM client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Provider = Literal["openai", "anthropic", "deepseek", "gemini", "grok", "mistral"]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: Provider
    env_key: str
    base_url: str
    default_model: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    provider: Provider
    model: str | None = None
    response: str | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""
    message: str
    provider: Provider
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

"""Error types for the serena-client library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ClientError:
    """Base error type for client operations."""

    message: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LanguageServerError(ClientError):
    """Error from language server operations."""

    language: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigurationError(ClientError):
    """Error in client configuration."""


@dataclass(frozen=True, slots=True)
class FileNotFoundError(ClientError):
    """File not found in project."""

    path: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolNotFoundError(ClientError):
    """Symbol not found in document or project."""

    symbol_name: str | None = None

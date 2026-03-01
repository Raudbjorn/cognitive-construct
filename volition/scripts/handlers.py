"""Handler Registry and Fallback Configuration for Volition.

Maps handler names to dispatcher functions and defines fallback chains.
This module decouples the plan executor from specific backend implementations.

See SPEC.md Section 5 Phase 4 (Tasks 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A handler function takes (action: str, inputs: dict) -> dict with status/data
HandlerFn = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class HandlerConfig:
    """Configuration for a registered handler."""

    name: str
    dispatch_fn: HandlerFn | None = None
    fallback_chain: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    risk_level: str = "LOW"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, HandlerConfig] = {}


def register_handler(config: HandlerConfig) -> None:
    """Register a handler configuration."""
    _REGISTRY[config.name] = config


def get_handler(name: str) -> HandlerConfig | None:
    """Get handler config by name."""
    return _REGISTRY.get(name)


def get_all_handlers() -> dict[str, HandlerConfig]:
    """Get all registered handlers."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Clear all registrations (for testing)."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

def register_defaults() -> None:
    """Register the default handler configurations.

    Called at startup; individual dispatch_fn's are wired in volition.py
    where the backend imports are available.
    """
    register_handler(HandlerConfig(
        name="code_edit",
        fallback_chain=["text_edit"],
        risk_level="MEDIUM",
    ))
    register_handler(HandlerConfig(
        name="text_edit",
        fallback_chain=[],
        risk_level="LOW",
    ))
    register_handler(HandlerConfig(
        name="llm_call",
        fallback_chain=[],
        risk_level="LOW",
    ))
    register_handler(HandlerConfig(
        name="web_search",
        fallback_chain=[],
        risk_level="LOW",
    ))
    register_handler(HandlerConfig(
        name="security",
        fallback_chain=[],
        requires_confirmation=True,
        risk_level="HIGH",
    ))

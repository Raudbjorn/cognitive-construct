"""Constitution management for session-based rules."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .config import MAX_RULES_PER_SESSION, SESSION_TTL_SECONDS


@dataclass
class ConstitutionEntry:
    """Entry storing rules and last access time."""

    rules: list[str] = field(default_factory=list)
    updated: float = 0.0


_constitution_map: dict[str, ConstitutionEntry] = {}
_lock = threading.Lock()
_cleanup_thread: threading.Timer | None = None


def update_constitution(session_id: str, rule: str) -> None:
    """Add a rule to a session's constitution."""
    if not session_id or not rule:
        return

    with _lock:
        entry = _constitution_map.get(session_id, ConstitutionEntry())

        if len(entry.rules) >= MAX_RULES_PER_SESSION:
            entry.rules.pop(0)

        entry.rules.append(rule)
        entry.updated = time.time()
        _constitution_map[session_id] = entry


def reset_constitution(session_id: str, rules: list[str]) -> None:
    """Replace all rules for a session."""
    if not session_id or not isinstance(rules, list):
        return

    with _lock:
        _constitution_map[session_id] = ConstitutionEntry(
            rules=rules[:MAX_RULES_PER_SESSION],
            updated=time.time(),
        )


def get_constitution(session_id: str) -> list[str]:
    """Get rules for a session."""
    with _lock:
        entry = _constitution_map.get(session_id)
        if not entry:
            return []

        entry.updated = time.time()
        return list(entry.rules)


def _cleanup() -> None:
    """Remove stale sessions to prevent unbounded memory growth."""
    global _cleanup_thread

    now = time.time()

    with _lock:
        stale_sessions = [
            session_id
            for session_id, entry in _constitution_map.items()
            if now - entry.updated > SESSION_TTL_SECONDS
        ]

        for session_id in stale_sessions:
            del _constitution_map[session_id]

    # Schedule next cleanup
    _schedule_cleanup()


def _schedule_cleanup() -> None:
    """Schedule the next cleanup."""
    global _cleanup_thread

    if _cleanup_thread:
        _cleanup_thread.cancel()

    _cleanup_thread = threading.Timer(SESSION_TTL_SECONDS, _cleanup)
    _cleanup_thread.daemon = True
    _cleanup_thread.start()


def start_cleanup() -> None:
    """Start the cleanup timer."""
    _schedule_cleanup()


def stop_cleanup() -> None:
    """Stop the cleanup timer."""
    global _cleanup_thread

    if _cleanup_thread:
        _cleanup_thread.cancel()
        _cleanup_thread = None


# Testing helpers
class _Testing:
    @staticmethod
    def get_map() -> dict[str, ConstitutionEntry]:
        return _constitution_map

    @staticmethod
    def cleanup() -> None:
        _cleanup()

    @staticmethod
    def clear() -> None:
        with _lock:
            _constitution_map.clear()


__testing = _Testing()

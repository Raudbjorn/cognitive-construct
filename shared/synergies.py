"""
Inter-Skill Synergies Module

Implements optional synergies between skills as specified in Requirements 8.1-8.5:
- Rhetoric → Encyclopedia: Context lookup during deliberation
- Volition → Inland Empire: Action logging for recall
- Encyclopedia → Inland Empire: Caching frequently accessed topics

These synergies operate transparently without requiring LLM orchestration.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from .validators import (
    SkillMessage,
    MessageType,
    SkillMessageBus,
    get_message_bus
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNERGY_CACHE_TTL = timedelta(hours=1)
SYNERGY_LOG_DIR = Path.home() / ".skills" / "synergy_logs"
INLAND_EMPIRE_CACHE_FILE = Path.home() / ".skills" / "encyclopedia_cache.json"


def ensure_synergy_dirs() -> None:
    """Ensure synergy directories exist."""
    SYNERGY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    INLAND_EMPIRE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synergy: Rhetoric → Encyclopedia (Context Lookup)
# Requirement 8.1: Rhetoric MAY consult Encyclopedia for relevant context
# ---------------------------------------------------------------------------

@dataclass
class ContextLookupResult:
    """Result from an Encyclopedia context lookup."""
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    source: str = "encyclopedia"
    cached: bool = False


def rhetoric_request_context(query: str, session_id: Optional[str] = None) -> Optional[ContextLookupResult]:
    """
    Rhetoric skill requests context from Encyclopedia for a deliberation question.

    This is a transparent synergy - if Encyclopedia handlers aren't registered
    or the lookup fails, it returns None and Rhetoric continues without context.
    """
    bus = get_message_bus()

    message = SkillMessage(
        source_skill="rhetoric",
        target_skill="encyclopedia",
        message_type=MessageType.CONTEXT_LOOKUP,
        payload={"query": query, "limit": 3},
        session_id=session_id
    )

    response = bus.send(message)

    if response and response.payload.get("success"):
        return ContextLookupResult(
            query=query,
            results=response.payload.get("results", []),
            source=response.payload.get("source", "encyclopedia"),
            cached=response.payload.get("cached", False)
        )
    return None


def encyclopedia_context_handler(message: SkillMessage) -> Optional[SkillMessage]:
    """
    Encyclopedia handler for context lookup requests from Rhetoric.

    This is registered when Encyclopedia skill initializes.
    Falls back to cached results if available.
    """
    if message.message_type != MessageType.CONTEXT_LOOKUP:
        return None

    query = message.payload.get("query", "")
    limit = message.payload.get("limit", 3)

    # Check cache first
    cached_results = _get_encyclopedia_cache(query)
    if cached_results:
        return message.create_response({
            "results": cached_results[:limit],
            "source": "encyclopedia_cache",
            "cached": True
        })

    # If no cache, return None - the actual Encyclopedia search
    # would be triggered by the Encyclopedia skill itself
    return None


# ---------------------------------------------------------------------------
# Synergy: Volition → Inland Empire (Action Logging)
# Requirement 8.2: Volition MAY log actions to Inland Empire for recall
# ---------------------------------------------------------------------------

@dataclass
class ActionLogEntry:
    """Entry for an action logged to Inland Empire."""
    action_type: str
    description: str
    timestamp: str
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def volition_log_action(
    action_type: str,
    description: str,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None
) -> bool:
    """
    Volition skill logs an action to Inland Empire for future recall.

    Returns True if the action was logged successfully, False otherwise.
    This is fire-and-forget - Volition doesn't wait for confirmation.
    """
    bus = get_message_bus()

    entry = ActionLogEntry(
        action_type=action_type,
        description=description,
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        metadata=metadata or {}
    )

    message = SkillMessage(
        source_skill="volition",
        target_skill="inland-empire",
        message_type=MessageType.ACTION_LOG,
        payload={
            "action_type": entry.action_type,
            "description": entry.description,
            "timestamp": entry.timestamp,
            "metadata": entry.metadata
        },
        session_id=session_id
    )

    # Fire-and-forget - we don't wait for response
    bus.send(message)

    # Also write to local log as backup
    _write_action_log(entry)

    return True


def inland_empire_action_handler(message: SkillMessage) -> Optional[SkillMessage]:
    """
    Inland Empire handler for action log events from Volition.

    Stores the action in memory for future consultation.
    """
    if message.message_type != MessageType.ACTION_LOG:
        return None

    # Extract action details
    action_type = message.payload.get("action_type", "unknown")
    description = message.payload.get("description", "")
    timestamp = message.payload.get("timestamp", datetime.now(timezone.utc).isoformat())
    metadata = message.payload.get("metadata", {})

    # Store in Inland Empire's memory (this would call the actual skill)
    # For now, we just acknowledge receipt
    _store_action_memory(action_type, description, timestamp, metadata)

    return message.create_response({"stored": True})


def _write_action_log(entry: ActionLogEntry) -> None:
    """Write action to local log file."""
    ensure_synergy_dirs()
    log_file = SYNERGY_LOG_DIR / f"actions_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps({
            "action_type": entry.action_type,
            "description": entry.description,
            "timestamp": entry.timestamp,
            "session_id": entry.session_id,
            "metadata": entry.metadata
        }) + "\n")


def _store_action_memory(
    action_type: str,
    description: str,
    timestamp: str,
    metadata: dict
) -> None:
    """Store action in Inland Empire's memory format."""
    # This would integrate with inland_empire.py's add_to_sql/add_to_graph
    # For now, we store in a simple format
    ensure_synergy_dirs()
    memory_file = SYNERGY_LOG_DIR / "action_memory.jsonl"

    with open(memory_file, "a") as f:
        f.write(json.dumps({
            "type": "action",
            "action_type": action_type,
            "content": description,
            "timestamp": timestamp,
            "metadata": metadata
        }) + "\n")


# ---------------------------------------------------------------------------
# Synergy: Encyclopedia → Inland Empire (Caching)
# Requirement 8.3: Encyclopedia MAY store frequently accessed topics in Inland Empire
# ---------------------------------------------------------------------------

def encyclopedia_cache_result(
    query: str,
    results: list[dict[str, Any]],
    session_id: Optional[str] = None,
    ttl_seconds: int = 3600
) -> bool:
    """
    Encyclopedia stores a search result in Inland Empire for caching.

    Returns True if cached successfully, False otherwise.
    """
    bus = get_message_bus()

    message = SkillMessage(
        source_skill="encyclopedia",
        target_skill="inland-empire",
        message_type=MessageType.CACHE_STORE,
        payload={
            "query": query,
            "results": results,
            "cached_at": datetime.now(timezone.utc).isoformat()
        },
        session_id=session_id,
        ttl_seconds=ttl_seconds
    )

    bus.send(message)

    # Also store locally
    _store_encyclopedia_cache(query, results, ttl_seconds)

    return True


def inland_empire_cache_handler(message: SkillMessage) -> Optional[SkillMessage]:
    """
    Inland Empire handler for cache store requests from Encyclopedia.
    """
    if message.message_type != MessageType.CACHE_STORE:
        return None

    query = message.payload.get("query", "")
    results = message.payload.get("results", [])
    ttl = message.ttl_seconds or 3600

    _store_encyclopedia_cache(query, results, ttl)

    return message.create_response({"cached": True})


def _store_encyclopedia_cache(query: str, results: list[dict], ttl_seconds: int) -> None:
    """Store Encyclopedia results in local cache."""
    ensure_synergy_dirs()

    # Load existing cache
    cache: dict[str, Any] = {}
    if INLAND_EMPIRE_CACHE_FILE.exists():
        try:
            with open(INLAND_EMPIRE_CACHE_FILE) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}

    # Clean expired entries
    now = datetime.now(timezone.utc)
    cache = {
        k: v for k, v in cache.items()
        if datetime.fromisoformat(v.get("expires_at", "2000-01-01")) > now
    }

    # Add new entry
    expires_at = now + timedelta(seconds=ttl_seconds)
    cache[query.lower()] = {
        "results": results,
        "cached_at": now.isoformat(),
        "expires_at": expires_at.isoformat()
    }

    # Write back
    with open(INLAND_EMPIRE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _get_encyclopedia_cache(query: str) -> Optional[list[dict]]:
    """Retrieve cached Encyclopedia results."""
    if not INLAND_EMPIRE_CACHE_FILE.exists():
        return None

    try:
        with open(INLAND_EMPIRE_CACHE_FILE) as f:
            cache = json.load(f)

        entry = cache.get(query.lower())
        if entry:
            expires_at = datetime.fromisoformat(entry.get("expires_at", "2000-01-01"))
            if expires_at > datetime.now(timezone.utc):
                return entry.get("results")
    except (json.JSONDecodeError, IOError, KeyError):
        pass

    return None


# ---------------------------------------------------------------------------
# Synergy Registration
# ---------------------------------------------------------------------------

def register_all_synergies() -> None:
    """
    Register all synergy handlers with the message bus.

    This should be called during skill initialization to enable
    transparent inter-skill communication.
    """
    bus = get_message_bus()

    # Encyclopedia handlers
    bus.register_handler("encyclopedia", encyclopedia_context_handler)

    # Inland Empire handlers
    bus.register_handler("inland-empire", inland_empire_action_handler)
    bus.register_handler("inland-empire", inland_empire_cache_handler)


def unregister_all_synergies() -> None:
    """
    Unregister all synergy handlers.

    Useful for testing or when synergies should be disabled.
    """
    bus = get_message_bus()
    bus.unregister_handlers("encyclopedia")
    bus.unregister_handlers("inland-empire")


# ---------------------------------------------------------------------------
# Synergy Status
# ---------------------------------------------------------------------------

def get_synergy_status() -> dict[str, Any]:
    """Get status of all synergies."""
    bus = get_message_bus()

    return {
        "enabled": bus.is_enabled,
        "synergies": {
            "rhetoric_to_encyclopedia": {
                "type": "context_lookup",
                "description": "Rhetoric requests context during deliberation"
            },
            "volition_to_inland_empire": {
                "type": "action_log",
                "description": "Volition logs actions for future recall"
            },
            "encyclopedia_to_inland_empire": {
                "type": "cache_store",
                "description": "Encyclopedia caches frequently accessed topics"
            }
        },
        "handlers_registered": {
            skill: len(handlers)
            for skill, handlers in bus._handlers.items()
        } if hasattr(bus, '_handlers') else {}
    }

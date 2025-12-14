from __future__ import annotations

"""
Shared event schema utilities used by the Inland Empire event store and
emergence runtime components.

This module defines:
- Event: Immutable event record for the append-only log
- EventType: Standard event type taxonomy by skill domain
- EventFilter: Subscription filter predicates
- StateSnapshot: Lightweight log state summary
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set
from uuid import UUID, uuid4


class EventType(str, Enum):
    """
    Standard event types by skill domain.
    Format: domain.entity.action

    Skills emit these typed events; membranes filter based on types.
    """
    # Encyclopedia events
    ENCYCLOPEDIA_SEARCH_STARTED = "encyclopedia.search.started"
    ENCYCLOPEDIA_SEARCH_COMPLETED = "encyclopedia.search.completed"
    ENCYCLOPEDIA_CACHE_HIT = "encyclopedia.cache.hit"
    ENCYCLOPEDIA_CACHE_MISS = "encyclopedia.cache.miss"
    ENCYCLOPEDIA_SOURCE_DEGRADED = "encyclopedia.source.degraded"
    ENCYCLOPEDIA_SOURCE_RESTORED = "encyclopedia.source.restored"

    # Rhetoric events
    RHETORIC_THOUGHT_RECORDED = "rhetoric.thought.recorded"
    RHETORIC_DELIBERATION_STARTED = "rhetoric.deliberation.started"
    RHETORIC_DELIBERATION_COMPLETED = "rhetoric.deliberation.completed"
    RHETORIC_PATTERN_DETECTED = "rhetoric.pattern.detected"
    RHETORIC_DECISION_MADE = "rhetoric.decision.made"

    # Inland Empire (Memory) events
    MEMORY_FACT_STORED = "memory.fact.stored"
    MEMORY_PATTERN_STORED = "memory.pattern.stored"
    MEMORY_CONTEXT_STORED = "memory.context.stored"
    MEMORY_CONSULTED = "memory.consulted"
    MEMORY_EVICTED = "memory.evicted"

    # Volition events
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    ACTION_CONFIRMED = "action.confirmed"
    ACTION_REJECTED = "action.rejected"

    # System events
    SYSTEM_SKILL_STARTED = "system.skill.started"
    SYSTEM_SKILL_STOPPED = "system.skill.stopped"
    SYSTEM_ERROR = "system.error"

    # Feedback events (for 7.4)
    FEEDBACK_USEFUL = "feedback.useful"
    FEEDBACK_NOT_USEFUL = "feedback.not_useful"
    FEEDBACK_PARTIAL = "feedback.partial"
    FEEDBACK_TIMEOUT = "feedback.timeout"


class EventValidationError(ValueError):
    """Raised when an event payload fails validation."""


def _ensure_utc(ts: datetime) -> datetime:
    """Normalize timestamps to UTC to simplify storage."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(slots=True)
class Event:
    """
    Canonical event representation for the emergence subsystem.
    """

    event_type: str
    source_skill: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    causation_id: Optional[UUID] = None
    correlation_id: Optional[str] = None
    version: int = 1

    def __post_init__(self) -> None:
        if not self.event_type or not isinstance(self.event_type, str):
            raise EventValidationError("event_type must be a non-empty string.")
        if not self.source_skill or not isinstance(self.source_skill, str):
            raise EventValidationError("source_skill must be a non-empty string.")
        self.timestamp = _ensure_utc(self.timestamp)
        if not isinstance(self.payload, dict):
            raise EventValidationError("payload must be a dictionary.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a JSON-friendly dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "source_skill": self.source_skill,
            "payload": self.payload,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "correlation_id": self.correlation_id,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize the event as JSON."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create an event from a serialized dictionary."""
        try:
            event_id = UUID(str(data["id"])) if data.get("id") else uuid4()
            timestamp = datetime.fromisoformat(data["timestamp"])
        except Exception as exc:  # pragma: no cover - defensive conversion
            raise EventValidationError(f"Invalid event serialization: {exc}") from exc

        payload = data.get("payload") or {}
        causation = data.get("causation_id")
        correlation = data.get("correlation_id")

        return cls(
            id=event_id,
            timestamp=_ensure_utc(timestamp),
            event_type=data["event_type"],
            source_skill=data["source_skill"],
            payload=dict(payload),
            causation_id=UUID(causation) if causation else None,
            correlation_id=correlation,
            version=int(data.get("version", 1)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Event":
        """Deserialize an event from JSON."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EventValidationError(f"Invalid event JSON: {exc}") from exc
        return cls.from_dict(payload)


@dataclass(slots=True)
class StateSnapshot:
    """Lightweight snapshot of the event log."""

    event_count: int
    last_event_id: Optional[UUID]
    last_event_timestamp: Optional[datetime]


class EventFilter:
    """
    Filter predicate for event subscription.

    Allows skills to specify which events they want to receive
    based on event type, source skill, and payload content.

    Usage:
        # Filter for all memory events
        filter = EventFilter.for_types(
            EventType.MEMORY_FACT_STORED,
            EventType.MEMORY_PATTERN_STORED
        )

        # Filter for events from a specific skill
        filter = EventFilter.from_skill("rhetoric")

        # Complex filter with payload predicate
        filter = EventFilter(
            event_types={EventType.RHETORIC_DECISION_MADE},
            payload_predicate=lambda p: p.get("action_required") is True
        )
    """

    def __init__(
        self,
        event_types: Optional[Set[EventType]] = None,
        source_skills: Optional[Set[str]] = None,
        payload_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Create an event filter.

        Args:
            event_types: Set of event types to match (None = all types)
            source_skills: Set of source skills to match (None = all skills)
            payload_predicate: Function(payload) -> bool for content filtering
        """
        self.event_types = event_types
        self.source_skills = source_skills
        self.payload_predicate = payload_predicate

    def matches(self, event: Event) -> bool:
        """Check if an event matches this filter."""
        # Check event type
        if self.event_types:
            # Handle both EventType enum and raw string event_type
            event_type_value = (
                event.event_type.value
                if isinstance(event.event_type, EventType)
                else event.event_type
            )
            type_values = {t.value for t in self.event_types}
            if event_type_value not in type_values:
                return False

        # Check source skill
        if self.source_skills and event.source_skill not in self.source_skills:
            return False

        # Check payload predicate
        if self.payload_predicate:
            try:
                if not self.payload_predicate(event.payload):
                    return False
            except Exception:
                # Payload predicate errors treated as non-match
                return False

        return True

    @classmethod
    def all(cls) -> "EventFilter":
        """Filter that matches all events."""
        return cls()

    @classmethod
    def for_types(cls, *types: EventType) -> "EventFilter":
        """Filter for specific event types."""
        return cls(event_types=set(types))

    @classmethod
    def from_skill(cls, skill: str) -> "EventFilter":
        """Filter for events from a specific skill."""
        return cls(source_skills={skill})

    @classmethod
    def from_skills(cls, *skills: str) -> "EventFilter":
        """Filter for events from multiple skills."""
        return cls(source_skills=set(skills))

    def __repr__(self) -> str:
        parts = []
        if self.event_types:
            parts.append(f"types={[t.value for t in self.event_types]}")
        if self.source_skills:
            parts.append(f"skills={list(self.source_skills)}")
        if self.payload_predicate:
            parts.append("predicate=<fn>")
        return f"EventFilter({', '.join(parts) or 'all'})"


def create_event(
    event_type: str | EventType,
    source_skill: str,
    payload: Dict[str, Any],
    causation_id: Optional[UUID] = None,
    correlation_id: Optional[str] = None,
) -> Event:
    """
    Factory function for creating events.

    Args:
        event_type: The type of event (string or EventType enum)
        source_skill: The skill emitting this event
        payload: Event-specific data
        causation_id: ID of the causing event (optional)
        correlation_id: Shared session/task ID (optional)

    Returns:
        A new Event instance

    Example:
        event = create_event(
            event_type=EventType.MEMORY_FACT_STORED,
            source_skill="inland-empire",
            payload={"fact": "user prefers dark mode", "confidence": 0.9}
        )
    """
    type_value = event_type.value if isinstance(event_type, EventType) else event_type
    return Event(
        event_type=type_value,
        source_skill=source_skill,
        payload=payload,
        causation_id=causation_id,
        correlation_id=correlation_id,
    )


"""
Membrane DSL for skill event boundaries.

A membrane defines the event contract for a skill:
- absorbs: event types the skill listens to, with optional filters
- emits: event types the skill can produce

Membranes can be defined programmatically or parsed from SKILL.md frontmatter.

Usage:
    # Programmatic definition
    membrane = Membrane(
        name="volition",
        absorbs=[
            AbsorbRule(EventType.RHETORIC_DECISION_MADE, predicate=lambda p: p.get("action_required")),
            AbsorbRule(EventType.ENCYCLOPEDIA_CACHE_MISS),
        ],
        emits={EventType.ACTION_STARTED, EventType.ACTION_COMPLETED, EventType.ACTION_FAILED},
    )

    # Check if event should be absorbed
    if membrane.should_absorb(event):
        skill.handle(event)

    # From SKILL.md frontmatter
    membrane = Membrane.from_yaml(yaml_block)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Union

from .events import Event, EventFilter, EventType


class Priority(str, Enum):
    """Priority levels for event absorption."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class AbsorbRule:
    """
    Rule for absorbing a specific event type.

    Attributes:
        event_type: The event type to absorb
        predicate: Optional filter function on the payload
        priority: Processing priority (high/normal/low)
    """
    event_type: Union[EventType, str]
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    priority: Priority = Priority.NORMAL

    def matches(self, event: Event) -> bool:
        """Check if this rule matches an event."""
        # Compare event types (handle both enum and string)
        rule_type = self.event_type.value if isinstance(self.event_type, EventType) else self.event_type
        event_type = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type

        if rule_type != event_type:
            return False

        if self.predicate:
            try:
                return self.predicate(event.payload)
            except Exception:
                return False
        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AbsorbRule":
        """
        Create an AbsorbRule from a dictionary (YAML parsed).

        Expected format:
            {"event_type": "rhetoric.decision.made", "filter": "action_required == true", "priority": "high"}
        """
        event_type = data.get("event_type", "")
        filter_expr = data.get("filter", "")
        priority_str = data.get("priority", "normal")

        predicate = None
        if filter_expr:
            predicate = _parse_filter_expression(filter_expr)

        priority = Priority(priority_str) if priority_str in [p.value for p in Priority] else Priority.NORMAL

        return cls(event_type=event_type, predicate=predicate, priority=priority)


@dataclass
class EmitSchema:
    """
    Schema definition for an emittable event type.

    Attributes:
        event_type: The event type this skill can emit
        schema: Optional payload schema (field -> type mapping)
    """
    event_type: Union[EventType, str]
    schema: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmitSchema":
        """Create from dictionary."""
        return cls(
            event_type=data.get("event_type", ""),
            schema=data.get("schema"),
        )


@dataclass
class RateLimits:
    """Rate limiting configuration for membrane operations."""
    max_absorb_per_minute: int = 100
    max_emit_per_minute: int = 50


@dataclass
class Membrane:
    """
    Defines the event boundary for a skill.

    A membrane acts as a filter that determines:
    - Which events flow INTO the skill (absorbs)
    - Which events can flow OUT of the skill (emits)
    - Rate limiting constraints
    """
    name: str
    absorbs: list[AbsorbRule] = field(default_factory=list)
    emits: Set[Union[EventType, str]] = field(default_factory=set)
    emit_schemas: list[EmitSchema] = field(default_factory=list)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    version: int = 1

    def should_absorb(self, event: Event) -> bool:
        """Check if this membrane should absorb an event."""
        for rule in self.absorbs:
            if rule.matches(event):
                return True
        return False

    def can_emit(self, event_type: Union[EventType, str]) -> bool:
        """Check if this membrane allows emitting an event type."""
        type_value = event_type.value if isinstance(event_type, EventType) else event_type
        for allowed in self.emits:
            allowed_value = allowed.value if isinstance(allowed, EventType) else allowed
            if allowed_value == type_value:
                return True
        return False

    def validate_emission(self, event: Event) -> bool:
        """Validate that an event can be emitted through this membrane."""
        return self.can_emit(event.event_type)

    def get_absorb_priority(self, event: Event) -> Optional[Priority]:
        """Get the priority for absorbing an event, or None if not absorbed."""
        for rule in self.absorbs:
            if rule.matches(event):
                return rule.priority
        return None

    def to_filter(self) -> EventFilter:
        """Convert absorb rules to an EventFilter."""
        if not self.absorbs:
            return EventFilter()

        event_types: Set[EventType] = set()
        for rule in self.absorbs:
            if isinstance(rule.event_type, EventType):
                event_types.add(rule.event_type)
            else:
                # Try to find matching EventType
                try:
                    event_types.add(EventType(rule.event_type))
                except ValueError:
                    pass

        return EventFilter(event_types=event_types if event_types else None)

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any], skill_name: str) -> "Membrane":
        """
        Parse a membrane definition from YAML data.

        Expected format in SKILL.md frontmatter:
            membrane:
              version: 1
              absorbs:
                - event_type: rhetoric.deliberation.completed
                  filter: "payload.action_required == true"
                  priority: high
              emits:
                - event_type: action.started
                  schema:
                    action_id: uuid
                    action_type: string
              rate_limits:
                max_absorb_per_minute: 100
                max_emit_per_minute: 50
        """
        version = yaml_data.get("version", 1)

        # Parse absorb rules
        absorbs = []
        for item in yaml_data.get("absorbs", []):
            if isinstance(item, dict):
                absorbs.append(AbsorbRule.from_dict(item))
            elif isinstance(item, str):
                # Simple string format: "event.type"
                absorbs.append(AbsorbRule(event_type=item))

        # Parse emit schemas
        emits: Set[str] = set()
        emit_schemas = []
        for item in yaml_data.get("emits", []):
            if isinstance(item, dict):
                schema = EmitSchema.from_dict(item)
                emit_schemas.append(schema)
                et = schema.event_type
                emits.add(et.value if isinstance(et, EventType) else et)
            elif isinstance(item, str):
                emits.add(item)

        # Parse rate limits
        rate_limits_data = yaml_data.get("rate_limits", {})
        rate_limits = RateLimits(
            max_absorb_per_minute=rate_limits_data.get("max_absorb_per_minute", 100),
            max_emit_per_minute=rate_limits_data.get("max_emit_per_minute", 50),
        )

        return cls(
            name=skill_name,
            absorbs=absorbs,
            emits=emits,
            emit_schemas=emit_schemas,
            rate_limits=rate_limits,
            version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format."""
        return {
            "version": self.version,
            "absorbs": [
                {
                    "event_type": r.event_type.value if isinstance(r.event_type, EventType) else r.event_type,
                    "priority": r.priority.value,
                    "has_predicate": r.predicate is not None,
                }
                for r in self.absorbs
            ],
            "emits": [
                e.value if isinstance(e, EventType) else e for e in self.emits
            ],
            "rate_limits": {
                "max_absorb_per_minute": self.rate_limits.max_absorb_per_minute,
                "max_emit_per_minute": self.rate_limits.max_emit_per_minute,
            },
        }


def _parse_filter_expression(expr: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Parse a simple filter expression into a predicate function.

    Supported expressions:
        - "key == value"          (equality)
        - "key == true/false"     (boolean)
        - "key in ['a', 'b']"     (membership)
        - "key != value"          (inequality)
        - "key"                   (truthy check)

    For security, we don't use eval() but parse the expression manually.
    """
    expr = expr.strip()

    # Handle "payload.key" by stripping "payload." prefix
    expr = re.sub(r"\bpayload\.", "", expr)

    # Equality: key == value
    eq_match = re.match(r"(\w+)\s*==\s*(.+)", expr)
    if eq_match:
        key, value = eq_match.groups()
        value = value.strip()
        # Parse value
        if value.lower() == "true":
            parsed_value: Any = True
        elif value.lower() == "false":
            parsed_value = False
        elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
            parsed_value = value[1:-1]
        elif value.isdigit():
            parsed_value = int(value)
        else:
            parsed_value = value

        return lambda p, k=key, v=parsed_value: p.get(k) == v

    # Inequality: key != value
    neq_match = re.match(r"(\w+)\s*!=\s*(.+)", expr)
    if neq_match:
        key, value = neq_match.groups()
        value = value.strip()
        if value.lower() == "true":
            parsed_value = True
        elif value.lower() == "false":
            parsed_value = False
        elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
            parsed_value = value[1:-1]
        else:
            parsed_value = value

        return lambda p, k=key, v=parsed_value: p.get(k) != v

    # Membership: key in ['a', 'b']
    in_match = re.match(r"(\w+)\s+in\s+\[(.+)\]", expr)
    if in_match:
        key, values_str = in_match.groups()
        # Parse list of values
        values = []
        for v in values_str.split(","):
            v = v.strip().strip("'\"")
            values.append(v)
        return lambda p, k=key, vs=values: p.get(k) in vs

    # Truthy check: just "key"
    if re.match(r"^\w+$", expr):
        return lambda p, k=expr: bool(p.get(k))

    # Default: always match
    return lambda p: True


# Pre-defined membranes for core skills
def create_volition_membrane() -> Membrane:
    """Create the default membrane for Volition skill."""
    return Membrane(
        name="volition",
        absorbs=[
            AbsorbRule(
                event_type=EventType.RHETORIC_DECISION_MADE,
                predicate=lambda p: p.get("action_required") is True,
                priority=Priority.HIGH,
            ),
            AbsorbRule(
                event_type=EventType.ENCYCLOPEDIA_CACHE_MISS,
                predicate=lambda p: p.get("query_type") in ["code_context", "library_docs"],
                priority=Priority.NORMAL,
            ),
        ],
        emits={
            EventType.ACTION_STARTED,
            EventType.ACTION_COMPLETED,
            EventType.ACTION_FAILED,
            EventType.ACTION_CONFIRMED,
            EventType.ACTION_REJECTED,
        },
    )


def create_rhetoric_membrane() -> Membrane:
    """Create the default membrane for Rhetoric skill."""
    return Membrane(
        name="rhetoric",
        absorbs=[
            AbsorbRule(
                event_type=EventType.MEMORY_CONSULTED,
                priority=Priority.NORMAL,
            ),
            AbsorbRule(
                event_type=EventType.ENCYCLOPEDIA_SEARCH_COMPLETED,
                priority=Priority.NORMAL,
            ),
        ],
        emits={
            EventType.RHETORIC_THOUGHT_RECORDED,
            EventType.RHETORIC_DELIBERATION_STARTED,
            EventType.RHETORIC_DELIBERATION_COMPLETED,
            EventType.RHETORIC_DECISION_MADE,
            EventType.RHETORIC_PATTERN_DETECTED,
        },
    )


def create_encyclopedia_membrane() -> Membrane:
    """Create the default membrane for Encyclopedia skill."""
    return Membrane(
        name="encyclopedia",
        absorbs=[
            AbsorbRule(
                event_type=EventType.RHETORIC_THOUGHT_RECORDED,
                predicate=lambda p: p.get("needs_research") is True,
                priority=Priority.NORMAL,
            ),
        ],
        emits={
            EventType.ENCYCLOPEDIA_SEARCH_STARTED,
            EventType.ENCYCLOPEDIA_SEARCH_COMPLETED,
            EventType.ENCYCLOPEDIA_CACHE_HIT,
            EventType.ENCYCLOPEDIA_CACHE_MISS,
            EventType.ENCYCLOPEDIA_SOURCE_DEGRADED,
            EventType.ENCYCLOPEDIA_SOURCE_RESTORED,
        },
    )


def create_inland_empire_membrane() -> Membrane:
    """Create the default membrane for Inland Empire skill."""
    return Membrane(
        name="inland-empire",
        absorbs=[
            AbsorbRule(
                event_type=EventType.ACTION_COMPLETED,
                priority=Priority.NORMAL,
            ),
            AbsorbRule(
                event_type=EventType.RHETORIC_PATTERN_DETECTED,
                priority=Priority.NORMAL,
            ),
            AbsorbRule(
                event_type=EventType.FEEDBACK_USEFUL,
                priority=Priority.HIGH,
            ),
            AbsorbRule(
                event_type=EventType.FEEDBACK_NOT_USEFUL,
                priority=Priority.HIGH,
            ),
        ],
        emits={
            EventType.MEMORY_FACT_STORED,
            EventType.MEMORY_PATTERN_STORED,
            EventType.MEMORY_CONTEXT_STORED,
            EventType.MEMORY_CONSULTED,
            EventType.MEMORY_EVICTED,
        },
    )


# Registry of skill membranes
SKILL_MEMBRANES: Dict[str, Membrane] = {
    "volition": create_volition_membrane(),
    "rhetoric": create_rhetoric_membrane(),
    "encyclopedia": create_encyclopedia_membrane(),
    "inland-empire": create_inland_empire_membrane(),
}


def get_membrane(skill_name: str) -> Optional[Membrane]:
    """Get the membrane for a skill by name."""
    return SKILL_MEMBRANES.get(skill_name)


def register_membrane(membrane: Membrane) -> None:
    """Register or update a skill membrane."""
    SKILL_MEMBRANES[membrane.name] = membrane

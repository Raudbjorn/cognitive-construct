"""Type definitions for the memory_libsql client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Entity:
    """An entity in the knowledge graph.

    Entities represent named objects with a type and associated observations.
    """

    name: str
    entity_type: str
    observations: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Relation:
    """A relation between two entities.

    Relations connect entities with a typed relationship.
    """

    source: str
    target: str
    relation_type: str


@dataclass(frozen=True, slots=True)
class KnowledgeGraph:
    """A knowledge graph containing entities and their relations."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Configuration for connecting to a LibSQL database."""

    url: str
    auth_token: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryError:
    """Error details for memory operations."""

    message: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

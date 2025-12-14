"""Type definitions for the memory_graph client."""

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

    Relations connect entities with a typed relationship in active voice.
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
class ObservationUpdate:
    """Request to add observations to an entity."""

    entity_name: str
    contents: list[str]


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Result of adding observations to an entity."""

    entity_name: str
    added_observations: list[str]


@dataclass(frozen=True, slots=True)
class ObservationDeletion:
    """Request to delete observations from an entity."""

    entity_name: str
    observations: list[str]


@dataclass(frozen=True, slots=True)
class GraphError:
    """Error details for knowledge graph operations."""

    message: str
    code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

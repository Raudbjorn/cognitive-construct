"""JSONL file-backed knowledge graph client implementation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .result import Err, Ok, Result
from .types import (
    Entity,
    GraphError,
    KnowledgeGraph,
    ObservationDeletion,
    ObservationResult,
    ObservationUpdate,
    Relation,
)

# === Constants ===

DEFAULT_MEMORY_FILE = "memory.jsonl"
MEMORY_FILE_ENV_VAR = "MEMORY_FILE_PATH"


# === Helper Functions ===


def get_default_memory_path() -> Path:
    """Get the default memory file path."""
    if env_path := os.environ.get(MEMORY_FILE_ENV_VAR):
        return Path(env_path).resolve()
    return Path.cwd() / DEFAULT_MEMORY_FILE


def _serialize_entity(entity: Entity) -> str:
    """Serialize an entity to a JSONL line."""
    return json.dumps({
        "type": "entity",
        "name": entity.name,
        "entityType": entity.entity_type,
        "observations": list(entity.observations),
    })


def _serialize_relation(relation: Relation) -> str:
    """Serialize a relation to a JSONL line."""
    return json.dumps({
        "type": "relation",
        "from": relation.source,
        "to": relation.target,
        "relationType": relation.relation_type,
    })


def _parse_line(line: str) -> Entity | Relation | None:
    """Parse a JSONL line into an Entity or Relation."""
    try:
        data = json.loads(line)
        if data.get("type") == "entity":
            return Entity(
                name=data["name"],
                entity_type=data["entityType"],
                observations=data.get("observations", []),
            )
        if data.get("type") == "relation":
            return Relation(
                source=data["from"],
                target=data["to"],
                relation_type=data["relationType"],
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# === Client Class ===


@dataclass
class KnowledgeGraphClient:
    """Client for JSONL file-backed knowledge graph memory.

    A simple knowledge graph implementation that stores entities and
    relations in a JSONL file for persistence.

    Usage:
        from memory_graph import KnowledgeGraphClient, Entity, Relation

        # Create client
        client = KnowledgeGraphClient()  # Uses MEMORY_FILE_PATH env or default

        # Create entities
        result = await client.create_entities([
            Entity(name="Python", entity_type="language", observations=["High-level"]),
        ])

        # Add observations
        result = await client.add_observations([
            ObservationUpdate(entity_name="Python", contents=["Popular for AI"]),
        ])

        # Create relations
        result = await client.create_relations([
            Relation(source="Python", target="AI", relation_type="used_for"),
        ])

        # Search
        result = await client.search_nodes("Python")
        if result.is_ok():
            print(result.value.entities)

        # Read full graph
        result = await client.read_graph()
    """

    file_path: Path | str | None = None
    _resolved_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize resolved file path."""
        if self.file_path:
            self._resolved_path = Path(self.file_path).resolve()
        else:
            self._resolved_path = get_default_memory_path()

    async def _load_graph(self) -> Result[KnowledgeGraph, GraphError]:
        """Load the knowledge graph from file."""
        try:
            if not self._resolved_path.exists():
                return Ok(KnowledgeGraph())

            content = self._resolved_path.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            entities: list[Entity] = []
            relations: list[Relation] = []

            for line in lines:
                item = _parse_line(line)
                if isinstance(item, Entity):
                    entities.append(item)
                elif isinstance(item, Relation):
                    relations.append(item)

            return Ok(KnowledgeGraph(entities=entities, relations=relations))

        except OSError as e:
            return Err(GraphError(f"Failed to load graph: {e}", code="IO_ERROR"))

    async def _save_graph(self, graph: KnowledgeGraph) -> Result[None, GraphError]:
        """Save the knowledge graph to file."""
        try:
            lines = [
                *[_serialize_entity(e) for e in graph.entities],
                *[_serialize_relation(r) for r in graph.relations],
            ]

            # Ensure parent directory exists
            self._resolved_path.parent.mkdir(parents=True, exist_ok=True)

            self._resolved_path.write_text("\n".join(lines), encoding="utf-8")
            return Ok(None)

        except OSError as e:
            return Err(GraphError(f"Failed to save graph: {e}", code="IO_ERROR"))

    # === Entity Operations ===

    async def create_entities(
        self,
        entities: list[Entity],
    ) -> Result[list[Entity], GraphError]:
        """Create new entities in the knowledge graph.

        Entities with names that already exist are skipped.
        Returns the list of actually created entities.
        """
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        existing_names = {e.name for e in graph.entities}

        new_entities = [e for e in entities if e.name not in existing_names]

        if new_entities:
            # Create mutable list for modification
            updated_entities = list(graph.entities) + new_entities
            updated_graph = KnowledgeGraph(
                entities=updated_entities,
                relations=list(graph.relations),
            )

            save_result = await self._save_graph(updated_graph)
            if save_result.is_err():
                return save_result  # type: ignore

        return Ok(new_entities)

    async def delete_entities(
        self,
        entity_names: list[str],
    ) -> Result[None, GraphError]:
        """Delete entities and their associated relations."""
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        names_to_delete = set(entity_names)

        filtered_entities = [e for e in graph.entities if e.name not in names_to_delete]
        filtered_relations = [
            r for r in graph.relations
            if r.source not in names_to_delete and r.target not in names_to_delete
        ]

        updated_graph = KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)
        return await self._save_graph(updated_graph)

    # === Observation Operations ===

    async def add_observations(
        self,
        updates: list[ObservationUpdate],
    ) -> Result[list[ObservationResult], GraphError]:
        """Add observations to existing entities.

        Only adds observations that don't already exist on the entity.
        """
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        entity_map = {e.name: e for e in graph.entities}
        results: list[ObservationResult] = []
        updated_entities: list[Entity] = []

        for update in updates:
            entity = entity_map.get(update.entity_name)
            if not entity:
                return Err(
                    GraphError(
                        f"Entity not found: {update.entity_name}",
                        code="NOT_FOUND",
                    )
                )

            existing_obs = set(entity.observations)
            new_obs = [c for c in update.contents if c not in existing_obs]

            # Create updated entity with new observations
            updated_entity = Entity(
                name=entity.name,
                entity_type=entity.entity_type,
                observations=list(entity.observations) + new_obs,
            )
            entity_map[entity.name] = updated_entity

            results.append(
                ObservationResult(
                    entity_name=update.entity_name,
                    added_observations=new_obs,
                )
            )

        # Rebuild entity list preserving order
        for entity in graph.entities:
            updated_entities.append(entity_map[entity.name])

        updated_graph = KnowledgeGraph(entities=updated_entities, relations=list(graph.relations))
        save_result = await self._save_graph(updated_graph)
        if save_result.is_err():
            return save_result  # type: ignore

        return Ok(results)

    async def delete_observations(
        self,
        deletions: list[ObservationDeletion],
    ) -> Result[None, GraphError]:
        """Delete specific observations from entities."""
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        entity_map = {e.name: e for e in graph.entities}

        for deletion in deletions:
            entity = entity_map.get(deletion.entity_name)
            if entity:
                obs_to_delete = set(deletion.observations)
                filtered_obs = [o for o in entity.observations if o not in obs_to_delete]
                entity_map[deletion.entity_name] = Entity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    observations=filtered_obs,
                )

        updated_entities = [entity_map[e.name] for e in graph.entities]
        updated_graph = KnowledgeGraph(entities=updated_entities, relations=list(graph.relations))
        return await self._save_graph(updated_graph)

    # === Relation Operations ===

    async def create_relations(
        self,
        relations: list[Relation],
    ) -> Result[list[Relation], GraphError]:
        """Create new relations between entities.

        Relations that already exist are skipped.
        """
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value

        existing_relations = {
            (r.source, r.target, r.relation_type)
            for r in graph.relations
        }

        new_relations = [
            r for r in relations
            if (r.source, r.target, r.relation_type) not in existing_relations
        ]

        if new_relations:
            updated_relations = list(graph.relations) + new_relations
            updated_graph = KnowledgeGraph(
                entities=list(graph.entities),
                relations=updated_relations,
            )

            save_result = await self._save_graph(updated_graph)
            if save_result.is_err():
                return save_result  # type: ignore

        return Ok(new_relations)

    async def delete_relations(
        self,
        relations: list[Relation],
    ) -> Result[None, GraphError]:
        """Delete specific relations from the graph."""
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value

        relations_to_delete = {
            (r.source, r.target, r.relation_type)
            for r in relations
        }

        filtered_relations = [
            r for r in graph.relations
            if (r.source, r.target, r.relation_type) not in relations_to_delete
        ]

        updated_graph = KnowledgeGraph(entities=list(graph.entities), relations=filtered_relations)
        return await self._save_graph(updated_graph)

    # === Graph Operations ===

    async def read_graph(self) -> Result[KnowledgeGraph, GraphError]:
        """Read the entire knowledge graph."""
        return await self._load_graph()

    async def search_nodes(
        self,
        query: str,
    ) -> Result[KnowledgeGraph, GraphError]:
        """Search for entities matching the query.

        Searches entity names, types, and observations.
        Returns matching entities and relations between them.
        """
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        query_lower = query.lower()

        # Filter entities
        filtered_entities = [
            e for e in graph.entities
            if (
                query_lower in e.name.lower()
                or query_lower in e.entity_type.lower()
                or any(query_lower in obs.lower() for obs in e.observations)
            )
        ]

        # Get relations between filtered entities
        filtered_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r for r in graph.relations
            if r.source in filtered_names and r.target in filtered_names
        ]

        return Ok(KnowledgeGraph(entities=filtered_entities, relations=filtered_relations))

    async def open_nodes(
        self,
        names: list[str],
    ) -> Result[KnowledgeGraph, GraphError]:
        """Open specific entities by name.

        Returns the named entities and relations between them.
        """
        graph_result = await self._load_graph()
        if graph_result.is_err():
            return graph_result  # type: ignore

        graph = graph_result.value
        names_set = set(names)

        filtered_entities = [e for e in graph.entities if e.name in names_set]
        filtered_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r for r in graph.relations
            if r.source in filtered_names and r.target in filtered_names
        ]

        return Ok(KnowledgeGraph(entities=filtered_entities, relations=filtered_relations))


# === Convenience Functions ===


async def create_client(
    file_path: Path | str | None = None,
) -> KnowledgeGraphClient:
    """Create a knowledge graph client.

    Convenience function for quick setup.

    Usage:
        client = await create_client()
        result = await client.read_graph()
    """
    return KnowledgeGraphClient(file_path=file_path)


async def search(
    query: str,
    file_path: Path | str | None = None,
) -> Result[KnowledgeGraph, GraphError]:
    """Search the knowledge graph (convenience function).

    Creates a client for single use.
    """
    client = KnowledgeGraphClient(file_path=file_path)
    return await client.search_nodes(query)


async def read_graph(
    file_path: Path | str | None = None,
) -> Result[KnowledgeGraph, GraphError]:
    """Read the entire knowledge graph (convenience function).

    Creates a client for single use.
    """
    client = KnowledgeGraphClient(file_path=file_path)
    return await client.read_graph()

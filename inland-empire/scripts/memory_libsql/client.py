"""LibSQL-backed memory client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import libsql_experimental as libsql

from .result import Err, Ok, Result
from .types import DatabaseConfig, Entity, KnowledgeGraph, MemoryError, Relation

# === Constants ===

DEFAULT_DB_URL = "file:./memory-tool.db"
DB_URL_ENV_VAR = "LIBSQL_URL"
DB_AUTH_TOKEN_ENV_VAR = "LIBSQL_AUTH_TOKEN"
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


# === Helper Functions ===


def get_database_config(
    url: str | None = None,
    auth_token: str | None = None,
) -> DatabaseConfig:
    """Get database configuration from parameters or environment."""
    resolved_url = url or os.environ.get(DB_URL_ENV_VAR) or DEFAULT_DB_URL
    resolved_token = auth_token or os.environ.get(DB_AUTH_TOKEN_ENV_VAR)
    return DatabaseConfig(url=resolved_url, auth_token=resolved_token)


def _validate_entity(entity: Entity) -> Result[None, MemoryError]:
    """Validate an entity before storage."""
    if not entity.name or not entity.name.strip():
        return Err(MemoryError("Entity name must be a non-empty string", code="INVALID_NAME"))
    if not entity.entity_type or not entity.entity_type.strip():
        return Err(
            MemoryError(
                f'Invalid entity type for entity "{entity.name}"',
                code="INVALID_TYPE",
            )
        )
    if not entity.observations:
        return Err(
            MemoryError(
                f'Entity "{entity.name}" must have at least one observation',
                code="NO_OBSERVATIONS",
            )
        )
    for obs in entity.observations:
        if not isinstance(obs, str) or not obs.strip():
            return Err(
                MemoryError(
                    f'Entity "{entity.name}" has invalid observations',
                    code="INVALID_OBSERVATION",
                )
            )
    return Ok(None)


def _clamp_limit(limit: int) -> int:
    """Clamp limit to valid range."""
    return min(max(1, limit), MAX_LIMIT)


# === Client Class ===


@dataclass
class MemoryClient:
    """Client for LibSQL-backed persistent memory.

    Usage:
        from memory_libsql import MemoryClient

        # Create client (uses LIBSQL_URL env var or defaults to local file)
        client = MemoryClient()
        await client.initialize()

        # Create entities
        result = await client.create_entities([
            Entity(name="Python", entity_type="language", observations=["High-level"]),
        ])
        if result.is_ok():
            print(f"Created {len(result.value)} entities")

        # Search
        result = await client.search_nodes("Python")
        if result.is_ok():
            for entity in result.value.entities:
                print(f"{entity.name}: {entity.observations}")

        # Cleanup
        await client.close()
    """

    url: str | None = None
    auth_token: str | None = None
    _conn: Any = field(init=False, repr=False, default=None)
    _config: DatabaseConfig = field(init=False, repr=False, default=None)  # type: ignore

    def __post_init__(self) -> None:
        """Initialize configuration."""
        self._config = get_database_config(self.url, self.auth_token)

    async def initialize(self) -> Result[None, MemoryError]:
        """Initialize the database connection and schema.

        Must be called before using other methods.
        """
        try:
            # Only pass auth_token if it's set (for remote Turso connections)
            if self._config.auth_token:
                self._conn = libsql.connect(
                    self._config.url,
                    auth_token=self._config.auth_token,
                )
            else:
                self._conn = libsql.connect(self._config.url)

            # Create tables
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_name) REFERENCES entities(name)
                )
            """)

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source) REFERENCES entities(name),
                    FOREIGN KEY (target) REFERENCES entities(name)
                )
            """)

            # Create indexes
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target)"
            )

            self._conn.commit()
            return Ok(None)

        except Exception as e:
            return Err(MemoryError(f"Database initialization failed: {e}", code="INIT_FAILED"))

    async def close(self) -> None:
        """Close the database connection."""
        import contextlib

        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def _check_connection(self) -> Result[None, MemoryError]:
        """Check if connection is initialized."""
        if not self._conn:
            return Err(
                MemoryError(
                    "Database not initialized. Call initialize() first.",
                    code="NOT_INITIALIZED",
                )
            )
        return Ok(None)

    # === Entity Operations ===

    async def create_entities(
        self,
        entities: list[Entity],
    ) -> Result[list[Entity], MemoryError]:
        """Create or update entities with observations.

        Existing entities are updated with new observations.
        """
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            created: list[Entity] = []

            for entity in entities:
                validation = _validate_entity(entity)
                if validation.is_err():
                    return validation  # type: ignore

                # Upsert entity
                existing = self._conn.execute(
                    "SELECT name FROM entities WHERE name = ?",
                    (entity.name,),
                ).fetchone()

                if existing:
                    self._conn.execute(
                        "UPDATE entities SET entity_type = ? WHERE name = ?",
                        (entity.entity_type, entity.name),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
                        (entity.name, entity.entity_type),
                    )

                # Replace observations
                self._conn.execute(
                    "DELETE FROM observations WHERE entity_name = ?",
                    (entity.name,),
                )

                for observation in entity.observations:
                    self._conn.execute(
                        "INSERT INTO observations (entity_name, content) VALUES (?, ?)",
                        (entity.name, observation),
                    )

                created.append(entity)

            self._conn.commit()
            return Ok(created)

        except Exception as e:
            return Err(MemoryError(f"Entity operation failed: {e}", code="ENTITY_ERROR"))

    async def get_entity(self, name: str) -> Result[Entity, MemoryError]:
        """Get a single entity by name."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            row = self._conn.execute(
                "SELECT name, entity_type FROM entities WHERE name = ?",
                (name,),
            ).fetchone()

            if not row:
                return Err(MemoryError(f"Entity not found: {name}", code="NOT_FOUND"))

            observations = self._conn.execute(
                "SELECT content FROM observations WHERE entity_name = ?",
                (name,),
            ).fetchall()

            return Ok(
                Entity(
                    name=row[0],
                    entity_type=row[1],
                    observations=[obs[0] for obs in observations],
                )
            )

        except Exception as e:
            return Err(MemoryError(f"Failed to get entity: {e}", code="GET_ERROR"))

    async def delete_entity(self, name: str) -> Result[None, MemoryError]:
        """Delete an entity and all associated data."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            existing = self._conn.execute(
                "SELECT name FROM entities WHERE name = ?",
                (name,),
            ).fetchone()

            if not existing:
                return Err(MemoryError(f"Entity not found: {name}", code="NOT_FOUND"))

            self._conn.execute("DELETE FROM observations WHERE entity_name = ?", (name,))
            self._conn.execute("DELETE FROM relations WHERE source = ? OR target = ?", (name, name))
            self._conn.execute("DELETE FROM entities WHERE name = ?", (name,))
            self._conn.commit()

            return Ok(None)

        except Exception as e:
            return Err(MemoryError(f"Failed to delete entity: {e}", code="DELETE_ERROR"))

    # === Relation Operations ===

    async def create_relations(
        self,
        relations: list[Relation],
    ) -> Result[list[Relation], MemoryError]:
        """Create relations between entities."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        if not relations:
            return Ok([])

        try:
            for relation in relations:
                self._conn.execute(
                    "INSERT INTO relations (source, target, relation_type) VALUES (?, ?, ?)",
                    (relation.source, relation.target, relation.relation_type),
                )

            self._conn.commit()
            return Ok(relations)

        except Exception as e:
            return Err(MemoryError(f"Failed to create relations: {e}", code="RELATION_ERROR"))

    async def delete_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
    ) -> Result[None, MemoryError]:
        """Delete a specific relation."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            cursor = self._conn.execute(
                "DELETE FROM relations WHERE source = ? AND target = ? AND relation_type = ?",
                (source, target, relation_type),
            )

            if cursor.rowcount == 0:
                return Err(
                    MemoryError(
                        f"Relation not found: {source} -> {target} ({relation_type})",
                        code="NOT_FOUND",
                    )
                )

            self._conn.commit()
            return Ok(None)

        except Exception as e:
            return Err(MemoryError(f"Failed to delete relation: {e}", code="DELETE_ERROR"))

    # === Search Operations ===

    async def search_entities(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> Result[list[Entity], MemoryError]:
        """Search entities by name, type, or observations."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            safe_limit = _clamp_limit(limit)
            normalized_query = f"%{query.replace(' ', '%').replace('_', '%').replace('-', '%')}%"

            rows = self._conn.execute(
                """
                SELECT DISTINCT
                    e.name,
                    e.entity_type,
                    CASE
                        WHEN e.name LIKE ? COLLATE NOCASE THEN 3
                        WHEN e.entity_type LIKE ? COLLATE NOCASE THEN 2
                        ELSE 1
                    END as relevance_score
                FROM entities e
                LEFT JOIN observations o ON e.name = o.entity_name
                WHERE e.name LIKE ? COLLATE NOCASE
                   OR e.entity_type LIKE ? COLLATE NOCASE
                   OR o.content LIKE ? COLLATE NOCASE
                ORDER BY relevance_score DESC, e.created_at DESC
                LIMIT ?
                """,
                (normalized_query,) * 5 + (safe_limit,),
            ).fetchall()

            entities: list[Entity] = []
            for row in rows:
                observations = self._conn.execute(
                    "SELECT content FROM observations WHERE entity_name = ?",
                    (row[0],),
                ).fetchall()

                entities.append(
                    Entity(
                        name=row[0],
                        entity_type=row[1],
                        observations=[obs[0] for obs in observations],
                    )
                )

            return Ok(entities)

        except Exception as e:
            return Err(MemoryError(f"Search failed: {e}", code="SEARCH_ERROR"))

    async def get_recent_entities(
        self,
        limit: int = DEFAULT_LIMIT,
    ) -> Result[list[Entity], MemoryError]:
        """Get most recently created entities."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        try:
            safe_limit = _clamp_limit(limit)

            rows = self._conn.execute(
                "SELECT name, entity_type FROM entities ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()

            entities: list[Entity] = []
            for row in rows:
                observations = self._conn.execute(
                    "SELECT content FROM observations WHERE entity_name = ?",
                    (row[0],),
                ).fetchall()

                entities.append(
                    Entity(
                        name=row[0],
                        entity_type=row[1],
                        observations=[obs[0] for obs in observations],
                    )
                )

            return Ok(entities)

        except Exception as e:
            return Err(MemoryError(f"Failed to get recent entities: {e}", code="QUERY_ERROR"))

    async def get_relations_for_entities(
        self,
        entities: list[Entity],
    ) -> Result[list[Relation], MemoryError]:
        """Get all relations involving the given entities."""
        conn_check = self._check_connection()
        if conn_check.is_err():
            return conn_check  # type: ignore

        if not entities:
            return Ok([])

        try:
            entity_names = [e.name for e in entities]
            placeholders = ",".join("?" * len(entity_names))

            rows = self._conn.execute(
                f"""
                SELECT source, target, relation_type
                FROM relations
                WHERE source IN ({placeholders})
                   OR target IN ({placeholders})
                """,
                tuple(entity_names + entity_names),
            ).fetchall()

            relations = [
                Relation(source=row[0], target=row[1], relation_type=row[2])
                for row in rows
            ]

            return Ok(relations)

        except Exception as e:
            return Err(MemoryError(f"Failed to get relations: {e}", code="QUERY_ERROR"))

    # === Graph Operations ===

    async def read_graph(self) -> Result[KnowledgeGraph, MemoryError]:
        """Get recent entities and their relations."""
        entities_result = await self.get_recent_entities()
        if entities_result.is_err():
            return entities_result  # type: ignore

        relations_result = await self.get_relations_for_entities(entities_result.value)
        if relations_result.is_err():
            return relations_result  # type: ignore

        return Ok(
            KnowledgeGraph(
                entities=entities_result.value,
                relations=relations_result.value,
            )
        )

    async def search_nodes(
        self,
        query: str,
        limit: int | None = None,
    ) -> Result[KnowledgeGraph, MemoryError]:
        """Search for entities and their relations."""
        if not query or not query.strip():
            return Err(MemoryError("Query cannot be empty", code="INVALID_QUERY"))

        entities_result = await self.search_entities(query, limit or DEFAULT_LIMIT)
        if entities_result.is_err():
            return entities_result  # type: ignore

        if not entities_result.value:
            return Ok(KnowledgeGraph())

        relations_result = await self.get_relations_for_entities(entities_result.value)
        if relations_result.is_err():
            return relations_result  # type: ignore

        return Ok(
            KnowledgeGraph(
                entities=entities_result.value,
                relations=relations_result.value,
            )
        )


# === Convenience Functions ===


async def create_client(
    url: str | None = None,
    auth_token: str | None = None,
) -> Result[MemoryClient, MemoryError]:
    """Create and initialize a memory client.

    Convenience function for quick setup.

    Usage:
        result = await create_client()
        if result.is_ok():
            client = result.value
            # ... use client ...
            await client.close()
    """
    client = MemoryClient(url=url, auth_token=auth_token)
    init_result = await client.initialize()
    if init_result.is_err():
        return init_result  # type: ignore
    return Ok(client)

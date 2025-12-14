"""
memory_libsql - Python client library for LibSQL-backed persistent memory

A knowledge graph memory system using LibSQL for persistence with
optimized text search and relevance ranking.

Usage:
    from memory_libsql import MemoryClient, Entity, Relation, create_client

    # Quick setup with convenience function
    result = await create_client()
    if result.is_err():
        print(f"Error: {result.error.message}")
        return

    client = result.value

    # Create entities
    result = await client.create_entities([
        Entity(
            name="Python",
            entity_type="programming_language",
            observations=["High-level", "Dynamically typed", "Popular for AI/ML"],
        ),
        Entity(
            name="Rust",
            entity_type="programming_language",
            observations=["Systems language", "Memory safe", "Fast"],
        ),
    ])

    # Create relations
    await client.create_relations([
        Relation(source="Python", target="Rust", relation_type="compared_to"),
    ])

    # Search
    result = await client.search_nodes("programming")
    if result.is_ok():
        graph = result.value
        for entity in graph.entities:
            print(f"{entity.name} ({entity.entity_type})")
            for obs in entity.observations:
                print(f"  - {obs}")

    # Cleanup
    await client.close()

Environment Variables:
    LIBSQL_URL: Database URL (default: file:./memory-tool.db)
    LIBSQL_AUTH_TOKEN: Authentication token for Turso/remote LibSQL
"""

from .client import MemoryClient, create_client, get_database_config
from .result import Err, Ok, Result
from .types import DatabaseConfig, Entity, KnowledgeGraph, MemoryError, Relation

__all__ = [
    # Client
    "MemoryClient",
    "create_client",
    "get_database_config",
    # Types
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "DatabaseConfig",
    "MemoryError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

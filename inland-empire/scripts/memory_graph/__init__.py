"""
memory_graph - Python client library for JSONL file-backed knowledge graph memory

A simple knowledge graph implementation that stores entities and
relations in a JSONL file for lightweight persistence.

Usage:
    from memory_graph import (
        KnowledgeGraphClient,
        Entity,
        Relation,
        ObservationUpdate,
        search,
        read_graph,
    )

    # Create client
    client = KnowledgeGraphClient()  # Uses MEMORY_FILE_PATH env or default

    # Create entities
    result = await client.create_entities([
        Entity(
            name="Python",
            entity_type="programming_language",
            observations=["High-level language", "Dynamically typed"],
        ),
        Entity(
            name="Machine Learning",
            entity_type="field",
            observations=["Subset of AI", "Uses statistical methods"],
        ),
    ])

    # Add more observations to existing entity
    result = await client.add_observations([
        ObservationUpdate(
            entity_name="Python",
            contents=["Popular for ML", "Has extensive libraries"],
        ),
    ])

    # Create relations (use active voice)
    result = await client.create_relations([
        Relation(source="Python", target="Machine Learning", relation_type="used_in"),
    ])

    # Search by query
    result = await client.search_nodes("Python")
    if result.is_ok():
        graph = result.value
        for entity in graph.entities:
            print(f"{entity.name} ({entity.entity_type})")
            for obs in entity.observations:
                print(f"  - {obs}")

    # Read full graph
    result = await client.read_graph()

    # Or use convenience functions
    result = await search("machine learning")
    result = await read_graph()

Environment Variables:
    MEMORY_FILE_PATH: Path to the JSONL memory file (default: ./memory.jsonl)
"""

from .client import (
    KnowledgeGraphClient,
    create_client,
    get_default_memory_path,
    read_graph,
    search,
)
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

__all__ = [
    # Client
    "KnowledgeGraphClient",
    "create_client",
    "get_default_memory_path",
    "search",
    "read_graph",
    # Types
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "ObservationUpdate",
    "ObservationResult",
    "ObservationDeletion",
    "GraphError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

"""
cgcli — Code graph CLI tool

Parse, index, and query code relationships across 17+ languages.
Uses SurrealDB embedded (zero-server) for graph storage and vector search.

Usage:
    from cgcli import CodeGraphClient

    client = CodeGraphClient()
    await client.connect()

    # Index a repository
    result = await client.index_repository("/path/to/repo")

    # Search for functions
    result = await client.find_function("process_data")
    if result.is_ok():
        for func in result.value:
            print(f"{func.name} at {func.file_path}:{func.line_number}")

    # Find who calls a function
    result = await client.who_calls("process_data")

    # Semantic vector search (requires embeddings extra)
    result = await client.vector_search("data processing utilities")

    await client.close()
"""

from .client import CodeGraphClient
from .database import DatabaseConfig, DatabaseManager
from ._types import (
    CallInfo,
    ClassHierarchy,
    ClassInfo,
    CodeGraphError,
    EXTENSION_TO_LANGUAGE,
    FunctionInfo,
    ImportInfo,
    Language,
    RelatedCodeResult,
    RepositoryInfo,
    SearchResult,
    VariableInfo,
)
from .result import Result, Ok, Err


def __getattr__(name: str):
    """Lazy-load IndexConfig to avoid importing tree_sitter at module level."""
    if name == "IndexConfig":
        from .indexer.config import IndexConfig
        return IndexConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeGraphClient",
    "DatabaseConfig",
    "DatabaseManager",
    "IndexConfig",
    "FunctionInfo",
    "ClassInfo",
    "VariableInfo",
    "CallInfo",
    "ImportInfo",
    "ClassHierarchy",
    "SearchResult",
    "RelatedCodeResult",
    "RepositoryInfo",
    "CodeGraphError",
    "Language",
    "EXTENSION_TO_LANGUAGE",
    "Result",
    "Ok",
    "Err",
]

__version__ = "2.0.0"

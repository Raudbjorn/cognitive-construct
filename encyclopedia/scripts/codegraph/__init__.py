"""
codegraph - Python client library for code graph analysis via Neo4j

A clean interface to CodeGraphContext functionality for indexing code
into a Neo4j graph database and querying code relationships.

Supports analysis of: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Ruby

Usage:
    from codegraph import CodeGraphClient

    # Create client (uses NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars)
    client = CodeGraphClient()

    # Or configure explicitly
    client = CodeGraphClient(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
    )

    # Connect
    result = client.connect()
    if result.is_err():
        print(f"Failed: {result.error.message}")
        exit(1)

    # Index a repository
    result = await client.index_repository("/path/to/repo")

    # Search for functions
    result = client.find_function("process_data")
    if result.is_ok():
        for func in result.value:
            print(f"{func.name} at {func.file_path}:{func.line_number}")

    # Find who calls a function
    result = client.who_calls("process_data")

    # Find what a function calls
    result = client.what_calls("main")

    # Get class hierarchy
    result = client.class_hierarchy("MyClass")
    if result.is_ok():
        hier = result.value
        print(f"Parents: {[p.name for p in hier.parent_classes]}")
        print(f"Children: {[c.name for c in hier.child_classes]}")

    # Find dead code
    result = client.find_dead_code()

    # Find complex functions
    result = client.most_complex_functions(limit=10)

    # List indexed repositories
    result = client.list_repositories()

    # Clean up
    client.close()
"""

from .client import CodeGraphClient
from .database import DatabaseManager, Neo4jConfig
from .types import (
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

__all__ = [
    "CodeGraphClient",
    "DatabaseManager",
    "Neo4jConfig",
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

__version__ = "1.0.0"

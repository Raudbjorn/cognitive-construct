"""CodeGraph client library for code analysis via Neo4j graph database."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .database import DatabaseManager, Neo4jConfig
from .result import Err, Ok, Result
from .types import (
    CallInfo,
    ClassHierarchy,
    ClassInfo,
    CodeGraphError,
    FunctionInfo,
    ImportInfo,
    RelatedCodeResult,
    RepositoryInfo,
    SearchResult,
    VariableInfo,
)


def _parse_function(record: dict) -> FunctionInfo:
    """Parse a function from Neo4j record."""
    return FunctionInfo(
        name=record.get("name") or record.get("function_name", ""),
        file_path=record.get("file_path", ""),
        line_number=record.get("line_number", 0),
        source=record.get("source"),
        docstring=record.get("docstring"),
        args=record.get("args", []) or [],
        decorators=record.get("decorators", []) or [],
        is_dependency=record.get("is_dependency", False),
        cyclomatic_complexity=record.get("complexity") or record.get("cyclomatic_complexity"),
    )


def _parse_class(record: dict) -> ClassInfo:
    """Parse a class from Neo4j record."""
    return ClassInfo(
        name=record.get("name") or record.get("class_name", ""),
        file_path=record.get("file_path") or record.get("class_file_path", ""),
        line_number=record.get("line_number") or record.get("class_line_number", 0),
        source=record.get("source"),
        docstring=record.get("docstring"),
        bases=record.get("bases", []) or [],
        is_dependency=record.get("is_dependency", False),
    )


def _parse_variable(record: dict) -> VariableInfo:
    """Parse a variable from Neo4j record."""
    return VariableInfo(
        name=record.get("name", ""),
        file_path=record.get("file_path", ""),
        line_number=record.get("line_number", 0),
        value=record.get("value"),
        context=record.get("context"),
        is_dependency=record.get("is_dependency", False),
    )


@dataclass
class CodeGraphClient:
    """Client for code graph analysis operations.

    This client provides a clean interface to CodeGraphContext's functionality
    for indexing and querying code relationships stored in Neo4j.

    Usage:
        from codegraph import CodeGraphClient

        # Configure (or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars)
        client = CodeGraphClient()

        # Connect
        result = client.connect()
        if result.is_err():
            print(f"Connection failed: {result.error.message}")

        # Index a repository
        result = await client.index_repository("/path/to/repo")

        # Search for code
        result = client.find_function("process_data")
        if result.is_ok():
            for func in result.value:
                print(f"{func.name} at {func.file_path}:{func.line_number}")

        # Find callers
        result = client.who_calls("process_data")

        # Find class hierarchy
        result = client.class_hierarchy("MyClass")
    """

    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    _db_manager: DatabaseManager = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._db_manager = DatabaseManager()
        if self.neo4j_uri and self.neo4j_password:
            config = Neo4jConfig(
                uri=self.neo4j_uri,
                username=self.neo4j_username or "neo4j",
                password=self.neo4j_password,
            )
            self._db_manager.configure(config)

    def connect(self) -> Result[None, CodeGraphError]:
        """Connect to Neo4j database."""
        return self._db_manager.connect()

    def close(self) -> None:
        """Close database connection."""
        self._db_manager.close()

    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._db_manager.is_connected()

    # -------------------------------------------------------------------------
    # Indexing operations
    # -------------------------------------------------------------------------

    async def index_repository(
        self, path: str | Path, as_dependency: bool = False
    ) -> Result[dict, CodeGraphError]:
        """Index a repository or directory into the graph.

        Args:
            path: Path to repository or directory
            as_dependency: Whether to mark as dependency code

        Returns:
            Result containing index statistics or error
        """
        try:
            # Import the heavy dependencies only when needed
            from codegraphcontext.tools.graph_builder import GraphBuilder
            from codegraphcontext.core.jobs import JobManager

            path = Path(path).resolve()
            if not path.exists():
                return Err(CodeGraphError(f"Path does not exist: {path}"))

            loop = asyncio.get_event_loop()
            job_manager = JobManager()
            builder = GraphBuilder(self._db_manager, job_manager, loop)

            await builder.build_graph_from_path_async(path, as_dependency)

            return Ok({
                "path": str(path),
                "indexed": True,
            })

        except ImportError:
            return Err(
                CodeGraphError(
                    "codegraphcontext package required for indexing",
                    "Install with: pip install codegraphcontext",
                )
            )
        except Exception as e:
            return Err(CodeGraphError(f"Indexing failed: {e}"))

    def list_repositories(self) -> Result[list[RepositoryInfo], CodeGraphError]:
        """List all indexed repositories."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                result = session.run("""
                    MATCH (r:Repository)
                    RETURN r.name as name, r.path as path, r.is_dependency as is_dependency
                    ORDER BY r.name
                """)
                repos = [
                    RepositoryInfo(
                        name=r["name"],
                        path=r["path"],
                        is_dependency=r.get("is_dependency", False),
                    )
                    for r in result
                ]
                return Ok(repos)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def delete_repository(self, path: str | Path) -> Result[None, CodeGraphError]:
        """Delete a repository and all its contents from the graph."""
        try:
            path_str = str(Path(path).resolve())
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                session.run(
                    """
                    MATCH (r:Repository {path: $path})
                    OPTIONAL MATCH (r)-[:CONTAINS*]->(e)
                    DETACH DELETE r, e
                    """,
                    path=path_str,
                )
            return Ok(None)
        except Exception as e:
            return Err(CodeGraphError(f"Delete failed: {e}"))

    # -------------------------------------------------------------------------
    # Search operations
    # -------------------------------------------------------------------------

    def find_function(
        self, name: str, fuzzy: bool = False
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        """Find functions by name."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                if fuzzy:
                    search_term = f"name:{name}"
                    result = session.run(
                        """
                        CALL db.index.fulltext.queryNodes("code_search_index", $search_term)
                        YIELD node, score
                        WHERE node:Function
                        RETURN node.name as name, node.file_path as file_path,
                               node.line_number as line_number, node.source as source,
                               node.docstring as docstring, node.args as args,
                               node.decorators as decorators, node.is_dependency as is_dependency,
                               node.cyclomatic_complexity as complexity
                        ORDER BY score DESC
                        LIMIT 20
                        """,
                        search_term=search_term,
                    )
                else:
                    result = session.run(
                        """
                        CALL db.index.fulltext.queryNodes("code_search_index", $name)
                        YIELD node, score
                        WHERE node:Function AND node.name CONTAINS $name
                        RETURN node.name as name, node.file_path as file_path,
                               node.line_number as line_number, node.source as source,
                               node.docstring as docstring, node.args as args,
                               node.decorators as decorators, node.is_dependency as is_dependency,
                               node.cyclomatic_complexity as complexity
                        ORDER BY score DESC
                        LIMIT 20
                        """,
                        name=name,
                    )
                return Ok([_parse_function(dict(r)) for r in result])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    def find_class(
        self, name: str, fuzzy: bool = False
    ) -> Result[list[ClassInfo], CodeGraphError]:
        """Find classes by name."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                if fuzzy:
                    search_term = f"name:{name}"
                    result = session.run(
                        """
                        CALL db.index.fulltext.queryNodes("code_search_index", $search_term)
                        YIELD node, score
                        WHERE node:Class
                        RETURN node.name as name, node.file_path as file_path,
                               node.line_number as line_number, node.source as source,
                               node.docstring as docstring, node.bases as bases,
                               node.is_dependency as is_dependency
                        ORDER BY score DESC
                        LIMIT 20
                        """,
                        search_term=search_term,
                    )
                else:
                    result = session.run(
                        """
                        CALL db.index.fulltext.queryNodes("code_search_index", $name)
                        YIELD node, score
                        WHERE node:Class AND node.name CONTAINS $name
                        RETURN node.name as name, node.file_path as file_path,
                               node.line_number as line_number, node.source as source,
                               node.docstring as docstring, node.bases as bases,
                               node.is_dependency as is_dependency
                        ORDER BY score DESC
                        LIMIT 20
                        """,
                        name=name,
                    )
                return Ok([_parse_class(dict(r)) for r in result])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    def find_variable(self, name: str) -> Result[list[VariableInfo], CodeGraphError]:
        """Find variables by name."""
        try:
            import re as re_module

            driver = self._db_manager.get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (v:Variable)
                    WHERE v.name CONTAINS $name OR v.name =~ $regex
                    RETURN v.name as name, v.file_path as file_path,
                           v.line_number as line_number, v.value as value,
                           v.context as context, v.is_dependency as is_dependency
                    ORDER BY v.is_dependency ASC, v.name
                    LIMIT 20
                    """,
                    name=name,
                    regex=f"(?i).*{re_module.escape(name)}.*",
                )
                return Ok([_parse_variable(dict(r)) for r in result])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    # -------------------------------------------------------------------------
    # Relationship queries
    # -------------------------------------------------------------------------

    def who_calls(
        self, function_name: str, file_path: str | None = None
    ) -> Result[list[CallInfo], CodeGraphError]:
        """Find what functions call a specific function."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                if file_path:
                    result = session.run(
                        """
                        MATCH (caller:Function)-[call:CALLS]->(target:Function {name: $name, file_path: $path})
                        RETURN caller.name as caller_name, caller.file_path as caller_file_path,
                               caller.line_number as caller_line_number,
                               target.name as called_name, target.file_path as called_file_path,
                               call.line_number as call_line_number, call.args as args
                        ORDER BY caller.is_dependency ASC, caller.file_path
                        LIMIT 20
                        """,
                        name=function_name,
                        path=str(Path(file_path).resolve()),
                    )
                else:
                    result = session.run(
                        """
                        MATCH (caller:Function)-[call:CALLS]->(target:Function {name: $name})
                        RETURN caller.name as caller_name, caller.file_path as caller_file_path,
                               caller.line_number as caller_line_number,
                               target.name as called_name, target.file_path as called_file_path,
                               call.line_number as call_line_number, call.args as args
                        ORDER BY caller.is_dependency ASC, caller.file_path
                        LIMIT 20
                        """,
                        name=function_name,
                    )
                calls = [
                    CallInfo(
                        caller_name=r["caller_name"],
                        caller_file_path=r["caller_file_path"],
                        caller_line_number=r["caller_line_number"],
                        called_name=r["called_name"],
                        called_file_path=r["called_file_path"],
                        call_line_number=r["call_line_number"],
                        args=r.get("args") or [],
                    )
                    for r in result
                ]
                return Ok(calls)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def what_calls(
        self, function_name: str, file_path: str | None = None
    ) -> Result[list[CallInfo], CodeGraphError]:
        """Find what functions a specific function calls."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                if file_path:
                    result = session.run(
                        """
                        MATCH (caller:Function {name: $name, file_path: $path})-[call:CALLS]->(called:Function)
                        RETURN caller.name as caller_name, caller.file_path as caller_file_path,
                               caller.line_number as caller_line_number,
                               called.name as called_name, called.file_path as called_file_path,
                               call.line_number as call_line_number, call.args as args
                        ORDER BY called.is_dependency ASC, called.name
                        LIMIT 20
                        """,
                        name=function_name,
                        path=str(Path(file_path).resolve()),
                    )
                else:
                    result = session.run(
                        """
                        MATCH (caller:Function {name: $name})-[call:CALLS]->(called:Function)
                        RETURN caller.name as caller_name, caller.file_path as caller_file_path,
                               caller.line_number as caller_line_number,
                               called.name as called_name, called.file_path as called_file_path,
                               call.line_number as call_line_number, call.args as args
                        ORDER BY called.is_dependency ASC, called.name
                        LIMIT 20
                        """,
                        name=function_name,
                    )
                calls = [
                    CallInfo(
                        caller_name=r["caller_name"],
                        caller_file_path=r["caller_file_path"],
                        caller_line_number=r["caller_line_number"],
                        called_name=r["called_name"],
                        called_file_path=r["called_file_path"],
                        call_line_number=r["call_line_number"],
                        args=r.get("args") or [],
                    )
                    for r in result
                ]
                return Ok(calls)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def who_imports(self, module_name: str) -> Result[list[ImportInfo], CodeGraphError]:
        """Find what files import a specific module."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (file:File)-[imp:IMPORTS]->(module:Module)
                    WHERE module.name = $name OR module.full_import_name CONTAINS $name
                    RETURN file.name as file_name, file.path as file_path,
                           module.name as module_name, module.alias as alias,
                           file.is_dependency as is_dependency
                    ORDER BY file.is_dependency ASC, file.path
                    LIMIT 20
                    """,
                    name=module_name,
                )
                imports = [
                    ImportInfo(
                        file_name=r["file_name"],
                        file_path=r["file_path"],
                        module_name=r["module_name"],
                        alias=r.get("alias"),
                        is_dependency=r.get("is_dependency", False),
                    )
                    for r in result
                ]
                return Ok(imports)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def class_hierarchy(
        self, class_name: str, file_path: str | None = None
    ) -> Result[ClassHierarchy, CodeGraphError]:
        """Get class inheritance hierarchy."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                match_clause = (
                    "MATCH (child:Class {name: $name, file_path: $path})"
                    if file_path
                    else "MATCH (child:Class {name: $name})"
                )
                params = {"name": class_name}
                if file_path:
                    params["path"] = str(Path(file_path).resolve())

                # Get parents
                parents_result = session.run(
                    f"""
                    {match_clause}
                    MATCH (child)-[:INHERITS]->(parent:Class)
                    RETURN parent.name as name, parent.file_path as file_path,
                           parent.line_number as line_number, parent.docstring as docstring,
                           parent.is_dependency as is_dependency
                    """,
                    **params,
                )
                parents = [_parse_class(dict(r)) for r in parents_result]

                # Get children
                children_result = session.run(
                    f"""
                    {match_clause}
                    MATCH (grandchild:Class)-[:INHERITS]->(child)
                    RETURN grandchild.name as name, grandchild.file_path as file_path,
                           grandchild.line_number as line_number, grandchild.docstring as docstring,
                           grandchild.is_dependency as is_dependency
                    """,
                    **params,
                )
                children = [_parse_class(dict(r)) for r in children_result]

                # Get methods
                methods_result = session.run(
                    f"""
                    {match_clause}
                    MATCH (child)-[:CONTAINS]->(method:Function)
                    RETURN method.name as name, method.file_path as file_path,
                           method.line_number as line_number, method.args as args,
                           method.docstring as docstring, method.is_dependency as is_dependency
                    """,
                    **params,
                )
                methods = [_parse_function(dict(r)) for r in methods_result]

                return Ok(
                    ClassHierarchy(
                        class_name=class_name,
                        parent_classes=parents,
                        child_classes=children,
                        methods=methods,
                    )
                )
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def find_dead_code(
        self, exclude_decorators: list[str] | None = None
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        """Find potentially unused functions."""
        try:
            exclude = exclude_decorators or []
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (func:Function)
                    WHERE func.is_dependency = false
                      AND NOT func.name IN ['main', '__init__', '__main__', 'setup', 'run']
                      AND NOT func.name STARTS WITH 'test_'
                      AND ALL(d IN $exclude WHERE NOT d IN func.decorators)
                    WITH func
                    OPTIONAL MATCH (caller:Function)-[:CALLS]->(func)
                    WHERE caller.is_dependency = false
                    WITH func, count(caller) as caller_count
                    WHERE caller_count = 0
                    RETURN func.name as name, func.file_path as file_path,
                           func.line_number as line_number, func.docstring as docstring,
                           func.is_dependency as is_dependency
                    ORDER BY func.file_path, func.line_number
                    LIMIT 50
                    """,
                    exclude=exclude,
                )
                return Ok([_parse_function(dict(r)) for r in result])
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    def most_complex_functions(
        self, limit: int = 10
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        """Find the most complex functions by cyclomatic complexity."""
        try:
            driver = self._db_manager.get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (f:Function)
                    WHERE f.cyclomatic_complexity IS NOT NULL AND f.is_dependency = false
                    RETURN f.name as name, f.file_path as file_path,
                           f.cyclomatic_complexity as complexity, f.line_number as line_number
                    ORDER BY f.cyclomatic_complexity DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                )
                return Ok([_parse_function(dict(r)) for r in result])
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

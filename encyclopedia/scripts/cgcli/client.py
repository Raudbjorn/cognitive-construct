"""CodeGraph client library for code analysis via SurrealDB graph database."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .database import DatabaseConfig, DatabaseManager
from .result import Err, Ok, Result
from ._types import (
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
    return FunctionInfo(
        name=record.get("name") or record.get("function_name", ""),
        file_path=record.get("file_path") or record.get("path", ""),
        line_number=record.get("line_number", 0),
        source=record.get("source"),
        docstring=record.get("docstring"),
        args=record.get("args") or [],
        decorators=record.get("decorators") or [],
        is_dependency=record.get("is_dependency", False),
        cyclomatic_complexity=record.get("complexity") or record.get("cyclomatic_complexity"),
    )


def _parse_class(record: dict) -> ClassInfo:
    return ClassInfo(
        name=record.get("name") or record.get("class_name", ""),
        file_path=record.get("file_path") or record.get("path", ""),
        line_number=record.get("line_number") or record.get("class_line_number", 0),
        source=record.get("source"),
        docstring=record.get("docstring"),
        bases=record.get("bases") or [],
        is_dependency=record.get("is_dependency", False),
    )


def _parse_variable(record: dict) -> VariableInfo:
    return VariableInfo(
        name=record.get("name", ""),
        file_path=record.get("file_path") or record.get("path", ""),
        line_number=record.get("line_number", 0),
        value=record.get("value"),
        context=record.get("context"),
        is_dependency=record.get("is_dependency", False),
    )


def _first_result(result: Any) -> list[dict]:
    """Extract result set from SurrealDB query response.

    SurrealDB Python SDK returns:
    - list[dict] for SELECT/CREATE/UPDATE/UPSERT
    - a scalar for RETURN
    - None for LET/IF blocks
    Multi-statement queries return only the LAST statement's result.
    """
    if result is None:
        return []
    if isinstance(result, list):
        # Already a list of dicts (the normal case)
        return [r for r in result if isinstance(r, dict)]
    if isinstance(result, dict):
        return [result]
    return []


def _flat_scalar(val: Any) -> Any:
    """Unwrap single-element lists from SurrealDB graph traversal results."""
    if isinstance(val, list):
        if len(val) == 1:
            return val[0]
        return val
    return val


@dataclass
class CodeGraphClient:
    """Client for code graph analysis operations.

    Usage:
        from cgcli import CodeGraphClient

        client = CodeGraphClient()
        await client.connect()

        result = await client.index_repository("/path/to/repo")
        result = await client.find_function("process_data")
        result = await client.who_calls("process_data")

        await client.close()
    """

    db_url: str | None = None
    _db_manager: DatabaseManager = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._db_manager = DatabaseManager()
        if self.db_url:
            self._db_manager._config = DatabaseConfig(db_url=self.db_url)

    async def connect(
        self, config: DatabaseConfig | None = None
    ) -> Result[None, CodeGraphError]:
        return await self._db_manager.connect(config)

    async def close(self) -> None:
        await self._db_manager.close()

    def is_connected(self) -> bool:
        return self._db_manager.is_connected()

    # -------------------------------------------------------------------------
    # Indexing operations
    # -------------------------------------------------------------------------

    async def index_repository(
        self, path: str | Path, as_dependency: bool = False
    ) -> Result[dict, CodeGraphError]:
        try:
            from .indexer.graph_builder import GraphBuilder
            from .indexer.jobs import JobManager

            path = Path(path).resolve()
            if not path.exists():
                return Err(CodeGraphError(f"Path does not exist: {path}"))

            loop = asyncio.get_event_loop()
            job_manager = JobManager()
            builder = GraphBuilder(self._db_manager, job_manager, loop)

            await builder.build_graph_from_path_async(path, as_dependency)

            return Ok({"path": str(path), "indexed": True})

        except Exception as e:
            return Err(CodeGraphError(f"Indexing failed: {e}"))

    async def list_repositories(self) -> Result[list[RepositoryInfo], CodeGraphError]:
        try:
            result = await self._db_manager.query(
                "SELECT name, path, is_dependency FROM repository ORDER BY name;"
            )
            rows = _first_result(result)
            return Ok([
                RepositoryInfo(
                    name=r.get("name", ""),
                    path=r.get("path", ""),
                    is_dependency=r.get("is_dependency", False),
                )
                for r in rows
            ])
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def delete_repository(
        self, path: str | Path
    ) -> Result[None, CodeGraphError]:
        try:
            from .indexer.graph_builder import GraphBuilder
            from .indexer.jobs import JobManager

            loop = asyncio.get_event_loop()
            builder = GraphBuilder(self._db_manager, JobManager(), loop)
            deleted = await builder.delete_repository_from_graph(str(path))
            if not deleted:
                return Err(CodeGraphError(f"Repository not found: {path}"))
            return Ok(None)
        except Exception as e:
            return Err(CodeGraphError(f"Delete failed: {e}"))

    # -------------------------------------------------------------------------
    # Search operations
    # -------------------------------------------------------------------------

    async def find_function(
        self, name: str, fuzzy: bool = False
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        try:
            if fuzzy:
                result = await self._db_manager.query(
                    """
                    SELECT name, path AS file_path, line_number, source, docstring,
                           args, decorators, is_dependency, complexity,
                           search::score(1) AS score
                    FROM node
                    WHERE node_type = 'Function'
                        AND (name @1@ $query OR source @1@ $query OR docstring @1@ $query)
                    ORDER BY score DESC
                    LIMIT 20;
                    """,
                    {"query": name},
                )
            else:
                result = await self._db_manager.query(
                    """
                    SELECT name, path AS file_path, line_number, source, docstring,
                           args, decorators, is_dependency, complexity
                    FROM node
                    WHERE node_type = 'Function' AND name CONTAINS $name
                    ORDER BY is_dependency ASC, name
                    LIMIT 20;
                    """,
                    {"name": name},
                )
            return Ok([_parse_function(r) for r in _first_result(result)])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    async def find_class(
        self, name: str, fuzzy: bool = False
    ) -> Result[list[ClassInfo], CodeGraphError]:
        try:
            if fuzzy:
                result = await self._db_manager.query(
                    """
                    SELECT name, path AS file_path, line_number, source, docstring,
                           bases, is_dependency,
                           search::score(1) AS score
                    FROM node
                    WHERE node_type = 'Class'
                        AND (name @1@ $query OR source @1@ $query OR docstring @1@ $query)
                    ORDER BY score DESC
                    LIMIT 20;
                    """,
                    {"query": name},
                )
            else:
                result = await self._db_manager.query(
                    """
                    SELECT name, path AS file_path, line_number, source, docstring,
                           bases, is_dependency
                    FROM node
                    WHERE node_type = 'Class' AND name CONTAINS $name
                    ORDER BY is_dependency ASC, name
                    LIMIT 20;
                    """,
                    {"name": name},
                )
            return Ok([_parse_class(r) for r in _first_result(result)])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    async def find_variable(
        self, name: str
    ) -> Result[list[VariableInfo], CodeGraphError]:
        try:
            result = await self._db_manager.query(
                """
                SELECT name, path AS file_path, line_number, value,
                       context, is_dependency
                FROM node
                WHERE node_type = 'Variable' AND name CONTAINS $name
                ORDER BY is_dependency ASC, name
                LIMIT 20;
                """,
                {"name": name},
            )
            return Ok([_parse_variable(r) for r in _first_result(result)])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

    # -------------------------------------------------------------------------
    # Relationship queries
    # -------------------------------------------------------------------------

    async def who_calls(
        self, function_name: str, file_path: str | None = None
    ) -> Result[list[CallInfo], CodeGraphError]:
        try:
            if file_path:
                resolved = str(Path(file_path).resolve())
                result = await self._db_manager.query(
                    """
                    SELECT
                        <-calls<-node.name AS caller_name,
                        <-calls<-node.path AS caller_file_path,
                        <-calls<-node.line_number AS caller_line_number,
                        name AS called_name,
                        path AS called_file_path,
                        <-calls.line_number AS call_line_number,
                        <-calls.args AS args
                    FROM node
                    WHERE name = $name
                        AND node_type IN ['Function', 'Class']
                        AND path = $path
                    LIMIT 20;
                    """,
                    {"name": function_name, "path": resolved},
                )
            else:
                result = await self._db_manager.query(
                    """
                    SELECT
                        <-calls<-node.name AS caller_name,
                        <-calls<-node.path AS caller_file_path,
                        <-calls<-node.line_number AS caller_line_number,
                        name AS called_name,
                        path AS called_file_path,
                        <-calls.line_number AS call_line_number,
                        <-calls.args AS args
                    FROM node
                    WHERE name = $name
                        AND node_type IN ['Function', 'Class']
                    LIMIT 20;
                    """,
                    {"name": function_name},
                )

            rows = _first_result(result)
            calls: list[CallInfo] = []
            for r in rows:
                # Graph traversal returns arrays — flatten each call pair
                caller_names = r.get("caller_name") or []
                caller_paths = r.get("caller_file_path") or []
                caller_lines = r.get("caller_line_number") or []
                call_lines = r.get("call_line_number") or []
                call_args = r.get("args") or []

                if not isinstance(caller_names, list):
                    caller_names = [caller_names]
                if not isinstance(caller_paths, list):
                    caller_paths = [caller_paths]
                if not isinstance(caller_lines, list):
                    caller_lines = [caller_lines]
                if not isinstance(call_lines, list):
                    call_lines = [call_lines]
                if not isinstance(call_args, list):
                    call_args = [call_args]

                for i in range(len(caller_names)):
                    calls.append(CallInfo(
                        caller_name=caller_names[i] if i < len(caller_names) else "",
                        caller_file_path=caller_paths[i] if i < len(caller_paths) else "",
                        caller_line_number=caller_lines[i] if i < len(caller_lines) else 0,
                        called_name=r.get("called_name", ""),
                        called_file_path=r.get("called_file_path", ""),
                        call_line_number=call_lines[i] if i < len(call_lines) else 0,
                        args=call_args[i] if i < len(call_args) and isinstance(call_args[i], list) else [],
                    ))
            return Ok(calls)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def what_calls(
        self, function_name: str, file_path: str | None = None
    ) -> Result[list[CallInfo], CodeGraphError]:
        try:
            if file_path:
                resolved = str(Path(file_path).resolve())
                result = await self._db_manager.query(
                    """
                    SELECT
                        name AS caller_name,
                        path AS caller_file_path,
                        line_number AS caller_line_number,
                        ->calls->node.name AS called_name,
                        ->calls->node.path AS called_file_path,
                        ->calls.line_number AS call_line_number,
                        ->calls.args AS args
                    FROM node
                    WHERE name = $name
                        AND node_type IN ['Function', 'Class']
                        AND path = $path
                    LIMIT 20;
                    """,
                    {"name": function_name, "path": resolved},
                )
            else:
                result = await self._db_manager.query(
                    """
                    SELECT
                        name AS caller_name,
                        path AS caller_file_path,
                        line_number AS caller_line_number,
                        ->calls->node.name AS called_name,
                        ->calls->node.path AS called_file_path,
                        ->calls.line_number AS call_line_number,
                        ->calls.args AS args
                    FROM node
                    WHERE name = $name
                        AND node_type IN ['Function', 'Class']
                    LIMIT 20;
                    """,
                    {"name": function_name},
                )

            rows = _first_result(result)
            calls: list[CallInfo] = []
            for r in rows:
                called_names = r.get("called_name") or []
                called_paths = r.get("called_file_path") or []
                call_lines = r.get("call_line_number") or []
                call_args = r.get("args") or []

                if not isinstance(called_names, list):
                    called_names = [called_names]
                if not isinstance(called_paths, list):
                    called_paths = [called_paths]
                if not isinstance(call_lines, list):
                    call_lines = [call_lines]
                if not isinstance(call_args, list):
                    call_args = [call_args]

                for i in range(len(called_names)):
                    calls.append(CallInfo(
                        caller_name=r.get("caller_name", ""),
                        caller_file_path=r.get("caller_file_path", ""),
                        caller_line_number=r.get("caller_line_number", 0),
                        called_name=called_names[i] if i < len(called_names) else "",
                        called_file_path=called_paths[i] if i < len(called_paths) else "",
                        call_line_number=call_lines[i] if i < len(call_lines) else 0,
                        args=call_args[i] if i < len(call_args) and isinstance(call_args[i], list) else [],
                    ))
            return Ok(calls)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def who_imports(
        self, module_name: str
    ) -> Result[list[ImportInfo], CodeGraphError]:
        try:
            result = await self._db_manager.query(
                """
                SELECT
                    <-imports<-node.name AS file_name,
                    <-imports<-node.path AS file_path,
                    name AS module_name,
                    <-imports.alias AS alias,
                    <-imports<-node.is_dependency AS is_dependency
                FROM node
                WHERE node_type = 'Module'
                    AND (name = $name OR full_import_name CONTAINS $name)
                LIMIT 20;
                """,
                {"name": module_name},
            )

            rows = _first_result(result)
            imports: list[ImportInfo] = []
            for r in rows:
                file_names = r.get("file_name") or []
                file_paths = r.get("file_path") or []
                aliases = r.get("alias") or []
                is_deps = r.get("is_dependency") or []

                if not isinstance(file_names, list):
                    file_names = [file_names]
                if not isinstance(file_paths, list):
                    file_paths = [file_paths]
                if not isinstance(aliases, list):
                    aliases = [aliases]
                if not isinstance(is_deps, list):
                    is_deps = [is_deps]

                for i in range(len(file_names)):
                    imports.append(ImportInfo(
                        file_name=file_names[i] if i < len(file_names) else "",
                        file_path=file_paths[i] if i < len(file_paths) else "",
                        module_name=r.get("module_name", ""),
                        alias=aliases[i] if i < len(aliases) else None,
                        is_dependency=is_deps[i] if i < len(is_deps) else False,
                    ))
            return Ok(imports)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def class_hierarchy(
        self, class_name: str, file_path: str | None = None
    ) -> Result[ClassHierarchy, CodeGraphError]:
        try:
            path_filter = "AND path = $path" if file_path else ""
            params: dict[str, Any] = {"name": class_name}
            if file_path:
                params["path"] = str(Path(file_path).resolve())

            # Parents
            parents_result = await self._db_manager.query(
                f"""
                SELECT ->inherits->node.* AS parents FROM node
                WHERE name = $name AND node_type = 'Class' {path_filter};
                """,
                params,
            )
            parents_rows = _first_result(parents_result)
            parents: list[ClassInfo] = []
            for r in parents_rows:
                for p in (r.get("parents") or []):
                    if isinstance(p, dict):
                        parents.append(_parse_class(p))

            # Children
            children_result = await self._db_manager.query(
                f"""
                SELECT <-inherits<-node.* AS children FROM node
                WHERE name = $name AND node_type = 'Class' {path_filter};
                """,
                params,
            )
            children_rows = _first_result(children_result)
            children: list[ClassInfo] = []
            for r in children_rows:
                for c in (r.get("children") or []):
                    if isinstance(c, dict):
                        children.append(_parse_class(c))

            # Methods
            methods_result = await self._db_manager.query(
                f"""
                SELECT ->contains->node.* AS methods FROM node
                WHERE name = $name AND node_type = 'Class' {path_filter};
                """,
                params,
            )
            methods_rows = _first_result(methods_result)
            methods: list[FunctionInfo] = []
            for r in methods_rows:
                for m in (r.get("methods") or []):
                    if isinstance(m, dict) and m.get("node_type") == "Function":
                        methods.append(_parse_function(m))

            return Ok(ClassHierarchy(
                class_name=class_name,
                parent_classes=parents,
                child_classes=children,
                methods=methods,
            ))
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def find_dead_code(
        self, exclude_decorators: list[str] | None = None
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        try:
            result = await self._db_manager.query(
                """
                SELECT name, path AS file_path, line_number, docstring,
                       is_dependency, decorators
                FROM node
                WHERE node_type = 'Function'
                    AND is_dependency = false
                    AND name NOT IN ['main', '__init__', '__main__', 'setup', 'run']
                    AND !string::starts_with(name, 'test_')
                    AND count(<-calls) = 0
                ORDER BY path, line_number
                LIMIT 50;
                """
            )
            rows = _first_result(result)

            exclude = set(exclude_decorators or [])
            funcs: list[FunctionInfo] = []
            for r in rows:
                decs = r.get("decorators") or []
                if exclude and any(d in exclude for d in decs):
                    continue
                funcs.append(_parse_function(r))
            return Ok(funcs)
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def most_complex_functions(
        self, limit: int = 10
    ) -> Result[list[FunctionInfo], CodeGraphError]:
        try:
            result = await self._db_manager.query(
                """
                SELECT name, path AS file_path, complexity, line_number
                FROM node
                WHERE node_type = 'Function'
                    AND complexity != NONE
                    AND is_dependency = false
                ORDER BY complexity DESC
                LIMIT $limit;
                """,
                {"limit": limit},
            )
            return Ok([_parse_function(r) for r in _first_result(result)])
        except Exception as e:
            return Err(CodeGraphError(f"Query failed: {e}"))

    async def vector_search(
        self, query: str, limit: int = 10
    ) -> Result[list[SearchResult], CodeGraphError]:
        try:
            from .embeddings import encode_query, is_available

            if not is_available():
                return Err(CodeGraphError(
                    "sentence-transformers not installed",
                    "Run: pip install 'cgcli[embeddings]'",
                ))

            query_vec = encode_query(query)
            if query_vec is None:
                return Err(CodeGraphError("Failed to encode query"))

            result = await self._db_manager.query(
                """
                SELECT id, name, path AS file_path, line_number, node_type,
                       source, docstring, is_dependency,
                       vector::similarity::cosine(embedding, $query_vec) AS score
                FROM node
                WHERE embedding <|$k, 200|> $query_vec
                    AND is_searchable = true
                ORDER BY score DESC;
                """,
                {"query_vec": query_vec.tolist(), "k": limit},
            )

            rows = _first_result(result)
            return Ok([
                SearchResult(
                    name=r.get("name", ""),
                    file_path=r.get("file_path", ""),
                    line_number=r.get("line_number", 0),
                    search_type=r.get("node_type", ""),
                    relevance_score=r.get("score", 0.0),
                    source=r.get("source"),
                    docstring=r.get("docstring"),
                    is_dependency=r.get("is_dependency", False),
                )
                for r in rows
            ])
        except Exception as e:
            return Err(CodeGraphError(f"Search failed: {e}"))

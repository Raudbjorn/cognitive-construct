#!/usr/bin/env python3
"""
Inland Empire - Unified memory substrate for Claude skills.

Store facts, patterns, and context. Query with remember, consult, and stats commands.

Backend mapping:
    fact_memory    -> memory_libsql (graph entities/relations)
    pattern_memory -> mem0/openmemory (hosted API or self-hosted)
    context_memory -> memory_graph (JSONL session memory)

Usage:
    python inland-empire.py remember "User prefers verbose errors"
    python inland-empire.py remember "Auth flow has race conditions" --type pattern
    python inland-empire.py consult "user preferences" --depth deep
    python inland-empire.py stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# === Constants ===

VERSION = "1.0.0"
DEFAULT_STATE_DIR = Path.cwd()
STATE_DIR_ENV = "INLAND_EMPIRE_STATE_DIR"
EVENT_HOME_ENV = "INLAND_EMPIRE_EVENT_HOME"
CONTEXT_MEMORY_FILE = "session_memory.jsonl"


class MemoryType(str, Enum):
    """Memory type aliases."""

    FACT = "fact"
    PATTERN = "pattern"
    CONTEXT = "context"


class SearchDepth(str, Enum):
    """Search depth options."""

    SHALLOW = "shallow"
    DEEP = "deep"


# === Result Types ===


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Success case."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Err[E]:
    """Error case."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


Result = Ok[Any] | Err[Any]


@dataclass(frozen=True, slots=True)
class MemoryError:
    """Memory operation error."""

    message: str
    code: str | None = None
    backend: str | None = None


# === Backend Detection ===


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Configuration for memory backends."""

    # fact_memory (libsql)
    libsql_url: str | None = None
    libsql_auth_token: str | None = None
    libsql_available: bool = True  # Always available (local fallback)

    # pattern_memory (mem0)
    mem0_api_key: str | None = None
    postgres_url: str | None = None
    mem0_mode: str | None = None  # "hosted", "self-hosted", or None (disabled)

    # context_memory (jsonl)
    context_file: Path | None = None
    context_available: bool = True  # Always available

    # General
    state_dir: Path = field(default_factory=Path.cwd)


def detect_backends() -> BackendConfig:
    """Detect available backends from environment."""
    state_dir = Path(os.environ.get(STATE_DIR_ENV, ".")).resolve()

    # LibSQL config
    libsql_url = os.environ.get("LIBSQL_URL")
    libsql_auth_token = os.environ.get("LIBSQL_AUTH_TOKEN")

    # Mem0 config
    mem0_api_key = os.environ.get("MEM0_API_KEY")
    postgres_url = os.environ.get("POSTGRES_URL")

    # Determine mem0 mode
    mem0_mode: str | None = None
    if mem0_api_key:
        mem0_mode = "hosted"
    elif postgres_url:
        mem0_mode = "self-hosted"

    # Context file
    context_file = state_dir / CONTEXT_MEMORY_FILE

    return BackendConfig(
        libsql_url=libsql_url,
        libsql_auth_token=libsql_auth_token,
        libsql_available=True,
        mem0_api_key=mem0_api_key,
        postgres_url=postgres_url,
        mem0_mode=mem0_mode,
        context_file=context_file,
        context_available=True,
        state_dir=state_dir,
    )


# === Memory Entry Types ===


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A memory entry with normalized metadata."""

    origin: str  # fact, pattern, or context
    summary: str
    score: float | None = None
    observed_at: str | None = None
    backend: str | None = None  # fact_memory, pattern_memory, context_memory
    partial: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConsultMetadata:
    """Metadata for a consult operation."""

    requested_backends: list[str]
    completed_backends: list[str]
    timed_out_backends: list[str]
    partial: bool


# === Backend Clients ===


class FactMemoryBackend:
    """Backend for fact_memory using memory_libsql."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._client: Any = None

    async def initialize(self) -> Result:
        """Initialize the LibSQL client."""
        try:
            # Import here to avoid hard dependency
            try:
                from memory_libsql import MemoryClient
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from memory_libsql import MemoryClient

            self._client = MemoryClient(
                url=self.config.libsql_url,
                auth_token=self.config.libsql_auth_token,
            )
            result = await self._client.initialize()
            if result.is_err():
                self._client = None  # Reset on failure
                return Err(MemoryError(result.error.message, "INIT_FAILED", "fact_memory"))
            return Ok(None)
        except ImportError as e:
            self._client = None
            return Err(MemoryError(f"memory_libsql dependency missing: {e}", "IMPORT_ERROR", "fact_memory"))
        except Exception as e:
            self._client = None
            return Err(MemoryError(str(e), "INIT_FAILED", "fact_memory"))

    async def store(self, text: str) -> Result:
        """Store a fact as an entity observation."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "fact_memory"))

        try:
            try:
                from memory_libsql import Entity
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from memory_libsql import Entity

            # Create a generic "fact" entity with the observation
            entity = Entity(
                name=f"fact_{hash(text) % 10000:04d}",
                entity_type="fact",
                observations=[text],
            )
            result = await self._client.create_entities([entity])
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "fact_memory"))

            return Ok({
                "stored": True,
                "backend": "fact_memory",
                "entity_name": entity.name,
            })
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "fact_memory"))

    async def search(self, query: str, limit: int = 10) -> Result:
        """Search for facts."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "fact_memory"))

        try:
            result = await self._client.search_nodes(query, limit=limit)
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "fact_memory"))

            entries = []
            graph = result.value
            for entity in graph.entities:
                for obs in entity.observations:
                    entries.append(
                        MemoryEntry(
                            origin="fact",
                            summary=obs,
                            backend="fact_memory",
                            metadata={
                                "entity_name": entity.name,
                                "entity_type": entity.entity_type,
                            },
                        )
                    )
            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "fact_memory"))

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()


class PatternMemoryBackend:
    """Backend for pattern_memory using mem0/openmemory."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._client: Any = None

    async def initialize(self) -> Result:
        """Initialize the Mem0 client."""
        if not self.config.mem0_mode:
            return Err(MemoryError("mem0 not configured", "NOT_CONFIGURED", "pattern_memory"))

        try:
            # Try installed package first, then fall back to local
            try:
                from openmemory import Mem0Client
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent / "mem0"))
                from openmemory import Mem0Client

            self._client = Mem0Client(api_key=self.config.mem0_api_key)
            return Ok(None)
        except ImportError:
            return Err(MemoryError("openmemory not installed", "IMPORT_ERROR", "pattern_memory"))
        except Exception as e:
            return Err(MemoryError(str(e), "INIT_FAILED", "pattern_memory"))

    async def store(self, text: str) -> Result:
        """Store a pattern in mem0."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "pattern_memory"))

        try:
            result = await self._client.add(text, user_id="agent_subconscious")
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "pattern_memory"))

            return Ok({
                "stored": True,
                "backend": "pattern_memory",
                "mode": self.config.mem0_mode,
            })
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "pattern_memory"))

    async def search(self, query: str, limit: int = 10) -> Result:
        """Search for patterns."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "pattern_memory"))

        try:
            result = await self._client.search(query, filters={"user_id": "agent_subconscious"}, top_k=limit)
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "pattern_memory"))

            entries = []
            for memory in result.value.results:
                entries.append(
                    MemoryEntry(
                        origin="pattern",
                        summary=memory.memory,
                        score=memory.score,
                        observed_at=memory.created_at,
                        backend="pattern_memory",
                        metadata={
                            "id": memory.id,
                            "user_id": "agent_subconscious",
                            "created_at": memory.created_at,
                            "updated_at": memory.updated_at,
                            "mode": self.config.mem0_mode,
                        },
                    )
                )
            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "pattern_memory"))

    async def close(self) -> None:
        """Close the client (no-op for mem0)."""
        pass


class ContextMemoryBackend:
    """Backend for context_memory using JSONL file."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._file_path: Path | None = config.context_file

    async def initialize(self) -> Result:
        """Initialize the context memory (ensure directory exists)."""
        if not self._file_path:
            return Err(MemoryError("No context file configured", "NOT_CONFIGURED", "context_memory"))

        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            return Ok(None)
        except Exception as e:
            return Err(MemoryError(str(e), "INIT_FAILED", "context_memory"))

    async def store(self, text: str) -> Result:
        """Store context to JSONL file."""
        if not self._file_path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "context_memory"))

        try:
            entry = {
                "type": "context",
                "content": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with self._file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            return Ok({
                "stored": True,
                "backend": "context_memory",
                "file": str(self._file_path),
            })
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "context_memory"))

    async def search(self, query: str, limit: int = 10) -> Result:
        """Search context memory (simple substring matching)."""
        if not self._file_path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "context_memory"))

        try:
            if not self._file_path.exists():
                return Ok([])

            entries = []
            query_lower = query.lower()

            for line in self._file_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("content", "")
                    if query_lower in content.lower():
                        entries.append(
                            MemoryEntry(
                                origin="context",
                                summary=content,
                                observed_at=data.get("timestamp"),
                                backend="context_memory",
                                metadata={
                                    "type": data.get("type"),
                                },
                            )
                        )
                except json.JSONDecodeError:
                    continue

                if len(entries) >= limit:
                    break

            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "context_memory"))

    async def count(self) -> int:
        """Count context entries."""
        if not self._file_path or not self._file_path.exists():
            return 0
        return sum(1 for line in self._file_path.read_text(encoding="utf-8").splitlines() if line.strip())

    async def close(self) -> None:
        """Close the backend (no-op for JSONL)."""
        pass


# === Unified Memory Manager ===


class InlandEmpire:
    """Unified memory substrate manager."""

    def __init__(self) -> None:
        self.config = detect_backends()
        self.fact = FactMemoryBackend(self.config)
        self.pattern = PatternMemoryBackend(self.config)
        self.context = ContextMemoryBackend(self.config)
        self._initialized = False
        self._init_results: dict[str, Result] = {}

    async def initialize(self) -> dict[str, Result]:
        """Initialize all backends."""
        results: dict[str, Result] = {}

        # Always initialize fact and context (they have local fallbacks)
        results["fact_memory"] = await self.fact.initialize()
        results["context_memory"] = await self.context.initialize()

        # Only initialize pattern if configured
        if self.config.mem0_mode:
            results["pattern_memory"] = await self.pattern.initialize()

        self._init_results = results
        self._initialized = True
        return results

    def _get_init_error(self, backend_name: str) -> str | None:
        """Get the initialization error message for a backend, if any."""
        result = self._init_results.get(backend_name)
        if result and result.is_err():
            return result.error.message
        return None

    async def remember(self, text: str, memory_type: MemoryType | None = None) -> dict[str, Any]:
        """Store a memory.

        If no type specified, stores to fact_memory by default.
        """
        if not self._initialized:
            await self.initialize()

        # Default to fact if not specified
        target_type = memory_type or MemoryType.FACT

        backend_map = {
            MemoryType.FACT: ("fact_memory", self.fact),
            MemoryType.PATTERN: ("pattern_memory", self.pattern),
            MemoryType.CONTEXT: ("context_memory", self.context),
        }

        backend_info = backend_map.get(target_type)
        if not backend_info:
            return {
                "status": "error",
                "command": "remember",
                "error": {"message": f"Unknown memory type: {target_type}", "code": "INVALID_TYPE"},
            }

        backend_name, backend = backend_info

        # Check if backend failed to initialize
        init_error = self._get_init_error(backend_name)
        if init_error:
            return {
                "status": "error",
                "command": "remember",
                "error": {
                    "message": f"Backend not available: {init_error}",
                    "code": "BACKEND_UNAVAILABLE",
                    "backend": backend_name,
                },
            }

        result = await backend.store(text)

        if result.is_err():
            return {
                "status": "error",
                "command": "remember",
                "error": {
                    "message": result.error.message,
                    "code": result.error.code,
                    "backend": result.error.backend,
                },
            }

        return {
            "status": "ok",
            "command": "remember",
            "result": result.value,
        }

    async def consult(
        self,
        query: str,
        depth: SearchDepth = SearchDepth.SHALLOW,
        memory_type: MemoryType | None = None,
    ) -> dict[str, Any]:
        """Query stored memories."""
        if not self._initialized:
            await self.initialize()

        limit = 5 if depth == SearchDepth.SHALLOW else 20
        requested_backends: list[str] = []
        completed_backends: list[str] = []
        timed_out_backends: list[str] = []
        all_results: list[MemoryEntry] = []

        # Determine which backends to query
        backends_to_query: list[tuple[str, Any]] = []

        if memory_type is None:
            # Query all available backends
            backends_to_query.append(("fact_memory", self.fact))
            if self.config.mem0_mode:
                backends_to_query.append(("pattern_memory", self.pattern))
            backends_to_query.append(("context_memory", self.context))
        elif memory_type == MemoryType.FACT:
            backends_to_query.append(("fact_memory", self.fact))
        elif memory_type == MemoryType.PATTERN:
            if self.config.mem0_mode:
                backends_to_query.append(("pattern_memory", self.pattern))
            else:
                return {
                    "status": "error",
                    "command": "consult",
                    "error": {"message": "pattern_memory not configured", "code": "NOT_CONFIGURED"},
                }
        elif memory_type == MemoryType.CONTEXT:
            backends_to_query.append(("context_memory", self.context))

        requested_backends = [name for name, _ in backends_to_query]
        unavailable_backends: list[str] = []

        # Query each backend
        for backend_name, backend in backends_to_query:
            # Skip backends that failed to initialize
            init_error = self._get_init_error(backend_name)
            if init_error:
                unavailable_backends.append(backend_name)
                continue

            try:
                result = await asyncio.wait_for(backend.search(query, limit=limit), timeout=30.0)
                if result.is_ok():
                    all_results.extend(result.value)
                    completed_backends.append(backend_name)
                else:
                    timed_out_backends.append(backend_name)
            except asyncio.TimeoutError:
                timed_out_backends.append(backend_name)
            except Exception:
                timed_out_backends.append(backend_name)

        # Convert MemoryEntry dataclasses to dicts
        results_dict = []
        for entry in all_results:
            results_dict.append({
                "origin": entry.origin,
                "summary": entry.summary,
                "score": entry.score,
                "observed_at": entry.observed_at,
                "backend": entry.backend,
                "partial": entry.partial,
                "metadata": entry.metadata,
            })

        return {
            "status": "ok",
            "command": "consult",
            "result": {
                "query": query,
                "depth": depth.value,
                "results": results_dict,
                "metadata": {
                    "requested_backends": requested_backends,
                    "completed_backends": completed_backends,
                    "unavailable_backends": unavailable_backends,
                    "timed_out_backends": timed_out_backends,
                    "partial": len(timed_out_backends) > 0 or len(unavailable_backends) > 0,
                },
            },
        }

    async def stats(self) -> dict[str, Any]:
        """Get backend health and statistics."""
        if not self._initialized:
            await self.initialize()

        backends = {}

        # Fact memory stats
        fact_error = self._get_init_error("fact_memory")
        if fact_error:
            backends["fact_memory"] = {
                "status": "unavailable",
                "backend": "memory_libsql",
                "error": fact_error,
            }
        else:
            backends["fact_memory"] = {
                "status": "available",
                "backend": "memory_libsql",
                "url": self.config.libsql_url or "file:./memory-tool.db (local)",
                "remote": bool(self.config.libsql_auth_token),
            }

        # Pattern memory stats
        if self.config.mem0_mode:
            pattern_error = self._get_init_error("pattern_memory")
            if pattern_error:
                backends["pattern_memory"] = {
                    "status": "unavailable",
                    "backend": "mem0",
                    "mode": self.config.mem0_mode,
                    "error": pattern_error,
                }
            else:
                backends["pattern_memory"] = {
                    "status": "available",
                    "backend": "mem0",
                    "mode": self.config.mem0_mode,
                }
        else:
            backends["pattern_memory"] = {
                "status": "disabled",
                "reason": "MEM0_API_KEY or POSTGRES_URL not set",
            }

        # Context memory stats
        context_error = self._get_init_error("context_memory")
        if context_error:
            backends["context_memory"] = {
                "status": "unavailable",
                "backend": "jsonl",
                "error": context_error,
            }
        else:
            context_count = await self.context.count()
            backends["context_memory"] = {
                "status": "available",
                "backend": "jsonl",
                "file": str(self.config.context_file),
                "entries": context_count,
            }

        return {
            "status": "ok",
            "command": "stats",
            "result": {
                "version": VERSION,
                "state_dir": str(self.config.state_dir),
                "backends": backends,
            },
        }

    async def close(self) -> None:
        """Close all backends."""
        await self.fact.close()
        await self.pattern.close()
        await self.context.close()


# === CLI ===


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="inland-empire",
        description="Unified memory substrate. Store facts, patterns, and context.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # remember command
    remember_parser = subparsers.add_parser(
        "remember",
        help="Store a memory across configured backends",
    )
    remember_parser.add_argument("text", help="The memory text to store")
    remember_parser.add_argument(
        "--type",
        "-t",
        choices=["fact", "pattern", "context"],
        default="fact",
        help="Memory type (default: fact)",
    )

    # consult command
    consult_parser = subparsers.add_parser(
        "consult",
        help="Query stored memories",
    )
    consult_parser.add_argument("query", help="The query string")
    consult_parser.add_argument(
        "--depth",
        "-d",
        choices=["shallow", "deep"],
        default="shallow",
        help="Search depth (default: shallow)",
    )
    consult_parser.add_argument(
        "--type",
        "-t",
        choices=["fact", "pattern", "context"],
        default=None,
        help="Filter by memory type",
    )

    # stats command
    subparsers.add_parser(
        "stats",
        help="Display backend health and statistics",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Check configuration before running
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from shared.config_check import require_skill_config
        require_skill_config("inland-empire", output_format="json")
    except ImportError:
        pass  # shared module not available, skip check

    empire = InlandEmpire()

    try:
        if args.command == "remember":
            memory_type = MemoryType(args.type) if args.type else None
            result = await empire.remember(args.text, memory_type)

        elif args.command == "consult":
            depth = SearchDepth(args.depth)
            memory_type = MemoryType(args.type) if args.type else None
            result = await empire.consult(args.query, depth, memory_type)

        elif args.command == "stats":
            result = await empire.stats()

        else:
            result = {"status": "error", "error": {"message": f"Unknown command: {args.command}"}}

        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "ok" else 1

    finally:
        await empire.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

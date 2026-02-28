#!/usr/bin/env python3
"""
Inland Empire - Subconscious memory layer for the Cognitive Construct.

Absorbs observations, surfaces relevant memories, and builds associative
context across sessions.

Commands:
    remember  - Commit something to memory (auto-classifies type)
    consult   - Actively search stored memories
    surface   - Broad associative retrieval ("gut feeling")
    forget    - Selectively remove memories
    stats     - Backend health and memory statistics

Backend mapping (internal, not exposed to user):
    graph    -> memory_libsql (LibSQL/SQLite entities + relations)
    semantic -> openmemory/mem0 (hosted or self-hosted semantic search)
    session  -> JSONL file (append-only session notes)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# === Constants ===

VERSION = "2.1.0"
STATE_DIR_ENV = "INLAND_EMPIRE_STATE_DIR"
SESSION_FILE = "session_memory.jsonl"

# Voice layer (Layer 2) — Mercury diffusion LLM
VOICE_BASE_URL = "https://api.inceptionlabs.ai/v1"
VOICE_MODEL = "mercury-small"
VOICE_MAX_TOKENS = 150
VOICE_TEMPERATURE = 1.0
VOICE_SYSTEM_PROMPT = (
    "You are Inland Empire, the gut feeling. You speak in fragments, "
    "associations, and half-formed intuitions. You do not explain or "
    "summarize. You connect things that seem unrelated. You feel the "
    "shape of problems before you can name them. Speak as a voice from "
    "the subconscious \u2014 oblique, impressionistic, sometimes wrong, "
    "always pointing at something."
)

# Content classification signals
_CONTEXT_SIGNALS = re.compile(
    r"\b(currently|right now|this session|debugging|investigating|working on|"
    r"todo|in progress|next step|blocked on|waiting for)\b",
    re.IGNORECASE,
)
_PATTERN_SIGNALS = re.compile(
    r"\b(always|usually|tends to|pattern|recurring|every time|often|"
    r"prefers?|habit|keeps? happening|flaky|intermittent|race condition)\b",
    re.IGNORECASE,
)


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


# === Content Classification ===


def classify_memory(text: str) -> MemoryType:
    """Infer memory type from content via keyword heuristics."""
    if _CONTEXT_SIGNALS.search(text):
        return MemoryType.CONTEXT
    if _PATTERN_SIGNALS.search(text):
        return MemoryType.PATTERN
    return MemoryType.FACT


def content_hash(text: str) -> str:
    """Deterministic content-addressable hash for entity naming."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# === Duration Parsing ===

_DURATION_RE = re.compile(r"^(\d+)([dhms])$")


def parse_duration(s: str) -> timedelta | None:
    """Parse a duration string like '7d', '24h', '30m', '60s'."""
    m = _DURATION_RE.match(s.strip())
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    match unit:
        case "d":
            return timedelta(days=value)
        case "h":
            return timedelta(hours=value)
        case "m":
            return timedelta(minutes=value)
        case "s":
            return timedelta(seconds=value)
    return None


# === Backend Detection ===


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Configuration for memory backends."""

    libsql_url: str | None = None
    libsql_auth_token: str | None = None
    mem0_api_key: str | None = None
    postgres_url: str | None = None
    mem0_mode: str | None = None  # "hosted" or None
    session_file: Path | None = None
    state_dir: Path = field(default_factory=Path.cwd)
    inception_api_key: str | None = None


def detect_backends() -> BackendConfig:
    """Detect available backends from environment."""
    state_dir = Path(os.environ.get(STATE_DIR_ENV, ".")).resolve()

    mem0_api_key = os.environ.get("MEM0_API_KEY")
    postgres_url = os.environ.get("POSTGRES_URL")
    mem0_mode: str | None = None
    if mem0_api_key:
        mem0_mode = "hosted"

    return BackendConfig(
        libsql_url=os.environ.get("LIBSQL_URL"),
        libsql_auth_token=os.environ.get("LIBSQL_AUTH_TOKEN"),
        mem0_api_key=mem0_api_key,
        postgres_url=postgres_url,
        mem0_mode=mem0_mode,
        session_file=state_dir / SESSION_FILE,
        state_dir=state_dir,
        inception_api_key=os.environ.get("INCEPTION_API_KEY"),
    )


# === Memory Entry ===


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A normalized memory entry returned from any backend."""

    summary: str
    type: str  # fact, pattern, context
    score: float | None = None
    observed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# === Backend: Graph (LibSQL) ===


def _import_libsql_module(name: str) -> Any:
    """Import from memory_libsql, falling back to local path."""
    try:
        mod = __import__("memory_libsql", fromlist=[name])
        return getattr(mod, name)
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        mod = __import__("memory_libsql", fromlist=[name])
        return getattr(mod, name)


class GraphBackend:
    """Structured graph storage for facts and pattern fallback."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._client: Any = None

    async def initialize(self) -> Result:
        """Initialize the LibSQL client."""
        try:
            MemoryClient = _import_libsql_module("MemoryClient")
            self._client = MemoryClient(
                url=self.config.libsql_url,
                auth_token=self.config.libsql_auth_token,
            )
            result = await self._client.initialize()
            if result.is_err():
                self._client = None
                return Err(MemoryError(result.error.message, "INIT_FAILED", "graph"))
            return Ok(None)
        except ImportError as e:
            self._client = None
            return Err(MemoryError(f"memory_libsql missing: {e}", "IMPORT_ERROR", "graph"))
        except Exception as e:
            self._client = None
            return Err(MemoryError(str(e), "INIT_FAILED", "graph"))

    async def store(self, text: str, entity_type: str = "fact") -> Result:
        """Store text as an entity observation with content-addressable name."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "graph"))

        try:
            Entity = _import_libsql_module("Entity")
            name = f"{entity_type}_{content_hash(text)}"
            entity = Entity(name=name, entity_type=entity_type, observations=[text])
            result = await self._client.create_entities([entity])
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "graph"))
            return Ok({"stored": True, "entity_name": name})
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "graph"))

    async def search(
        self, query: str, limit: int = 10, entity_type: str | None = None
    ) -> Result:
        """Search graph entities, optionally filtered by entity_type."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "graph"))

        try:
            result = await self._client.search_nodes(query, limit=limit)
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "graph"))

            entries = []
            for entity in result.value.entities:
                if entity_type and entity.entity_type != entity_type:
                    continue
                for obs in entity.observations:
                    entries.append(
                        MemoryEntry(
                            summary=obs,
                            type=entity.entity_type,
                            metadata={"entity_name": entity.name},
                        )
                    )
            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "graph"))

    async def delete_matching(
        self, query: str, entity_type: str | None = None
    ) -> Result:
        """Delete entities matching query. Returns count deleted."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "graph"))

        try:
            result = await self._client.search_nodes(query, limit=50)
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "graph"))

            deleted = 0
            for entity in result.value.entities:
                if entity_type and entity.entity_type != entity_type:
                    continue
                del_result = await self._client.delete_entity(entity.name)
                if del_result.is_ok():
                    deleted += 1
            return Ok(deleted)
        except Exception as e:
            return Err(MemoryError(str(e), "DELETE_FAILED", "graph"))

    async def close(self) -> None:
        """Close the database connection."""
        if self._client:
            await self._client.close()


# === Backend: Semantic (Mem0) ===


class SemanticBackend:
    """Semantic memory via Mem0 API."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._client: Any = None

    async def initialize(self) -> Result:
        """Initialize the Mem0 client."""
        if not self.config.mem0_mode:
            return Err(MemoryError("Not configured", "NOT_CONFIGURED", "semantic"))
        if not self.config.mem0_api_key:
            return Err(MemoryError(
                "MEM0_API_KEY required for semantic backend",
                "MISSING_API_KEY", "semantic",
            ))

        try:
            try:
                from openmemory import Mem0Client
            except ImportError:
                sys.path.insert(0, str(Path(__file__).parent / "mem0"))
                from openmemory import Mem0Client

            self._client = Mem0Client(api_key=self.config.mem0_api_key)
            return Ok(None)
        except ImportError:
            return Err(MemoryError("openmemory not installed", "IMPORT_ERROR", "semantic"))
        except Exception as e:
            return Err(MemoryError(str(e), "INIT_FAILED", "semantic"))

    async def store(self, text: str) -> Result:
        """Store a pattern in Mem0."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "semantic"))

        try:
            result = await self._client.add(text, user_id="agent_subconscious")
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "semantic"))
            return Ok({"stored": True, "mode": self.config.mem0_mode})
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "semantic"))

    async def search(self, query: str, limit: int = 10) -> Result:
        """Semantic search for patterns."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "semantic"))

        try:
            result = await self._client.search(
                query,
                filters={"user_id": "agent_subconscious"},
                top_k=limit,
            )
            if result.is_err():
                return Err(MemoryError(result.error.message, result.error.code, "semantic"))

            entries = []
            for mem in result.value.results:
                entries.append(
                    MemoryEntry(
                        summary=mem.memory,
                        type="pattern",
                        score=mem.score,
                        observed_at=mem.created_at,
                        metadata={"id": mem.id, "mode": self.config.mem0_mode},
                    )
                )
            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "semantic"))

    async def delete_matching(self, query: str) -> Result:
        """Search for matching memories and delete them. Returns count deleted."""
        if not self._client:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "semantic"))

        try:
            search_result = await self._client.search(
                query,
                filters={"user_id": "agent_subconscious"},
                top_k=20,
            )
            if search_result.is_err():
                return Err(
                    MemoryError(
                        search_result.error.message, search_result.error.code, "semantic"
                    )
                )

            deleted = 0
            for mem in search_result.value.results:
                del_result = await self._client.delete(mem.id)
                if del_result.is_ok():
                    deleted += 1
            return Ok(deleted)
        except Exception as e:
            return Err(MemoryError(str(e), "DELETE_FAILED", "semantic"))

    async def close(self) -> None:
        """No-op for Mem0."""


# === Backend: Session (JSONL) ===


class SessionBackend:
    """Append-only session context in JSONL."""

    def __init__(self, config: BackendConfig) -> None:
        self._path: Path | None = config.session_file

    async def initialize(self) -> Result:
        """Ensure session directory exists."""
        if not self._path:
            return Err(MemoryError("No session file configured", "NOT_CONFIGURED", "session"))

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return Ok(None)
        except Exception as e:
            return Err(MemoryError(str(e), "INIT_FAILED", "session"))

    async def store(self, text: str) -> Result:
        """Append context to JSONL file."""
        if not self._path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "session"))

        try:
            entry = {
                "type": "context",
                "content": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return Ok({"stored": True})
        except Exception as e:
            return Err(MemoryError(str(e), "STORE_FAILED", "session"))

    async def search(self, query: str, limit: int = 10) -> Result:
        """Search session memory (substring matching, most recent first)."""
        if not self._path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "session"))

        try:
            if not self._path.exists():
                return Ok([])

            entries: list[MemoryEntry] = []
            query_lower = query.lower()
            lines = self._path.read_text(encoding="utf-8").splitlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("content", "")
                if query_lower in content.lower():
                    entries.append(
                        MemoryEntry(
                            summary=content,
                            type="context",
                            observed_at=data.get("timestamp"),
                        )
                    )
                    if len(entries) >= limit:
                        break

            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "session"))

    async def delete_matching(
        self,
        query: str | None = None,
        before: datetime | None = None,
    ) -> Result:
        """Delete matching session entries. Rewrites JSONL file.

        When both query and before are provided, uses AND semantics:
        an entry must match the query AND be older than the threshold.
        When only one is provided, that criterion alone applies.
        """
        if not self._path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "session"))

        try:
            if not self._path.exists():
                return Ok(0)

            lines = self._path.read_text(encoding="utf-8").splitlines()
            keep: list[str] = []
            deleted = 0
            query_lower = query.lower() if query else None

            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    keep.append(line)
                    continue

                should_delete = True

                # If query specified, content must match
                if query_lower:
                    content = data.get("content", "")
                    if query_lower not in content.lower():
                        should_delete = False

                # If before specified, entry must be old enough
                if should_delete and before:
                    ts = data.get("timestamp")
                    if ts:
                        try:
                            entry_time = datetime.fromisoformat(ts)
                            if entry_time >= before:
                                should_delete = False
                        except ValueError:
                            should_delete = False
                    else:
                        should_delete = False  # No timestamp, can't determine age

                if should_delete:
                    deleted += 1
                else:
                    keep.append(line)

            self._path.write_text(
                "\n".join(keep) + ("\n" if keep else ""),
                encoding="utf-8",
            )
            return Ok(deleted)
        except Exception as e:
            return Err(MemoryError(str(e), "DELETE_FAILED", "session"))

    async def all_entries(self, limit: int = 50) -> Result:
        """Return all session entries (most recent first)."""
        if not self._path:
            return Err(MemoryError("Not initialized", "NOT_INITIALIZED", "session"))

        try:
            if not self._path.exists():
                return Ok([])

            entries: list[MemoryEntry] = []
            lines = self._path.read_text(encoding="utf-8").splitlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entries.append(
                    MemoryEntry(
                        summary=data.get("content", ""),
                        type="context",
                        observed_at=data.get("timestamp"),
                    )
                )
                if len(entries) >= limit:
                    break

            return Ok(entries)
        except Exception as e:
            return Err(MemoryError(str(e), "SEARCH_FAILED", "session"))

    async def count(self) -> int:
        """Count session entries."""
        if not self._path or not self._path.exists():
            return 0
        return sum(
            1
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    async def close(self) -> None:
        """No-op for JSONL."""


# === Voice Layer (Layer 2) ===


class VoiceLayer:
    """Associative voice synthesis via Mercury diffusion LLM.

    Generates an oblique, impressionistic reading of surfaced associations.
    Degrades silently to None when INCEPTION_API_KEY is not set or the API
    is unreachable.
    """

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._available = bool(api_key)

    async def generate(
        self, associations: list[dict[str, Any]], context: str
    ) -> str | None:
        """Generate a voice reading. Returns None if unavailable or on error."""
        if not self._available or not associations:
            return None

        # Cap associations to avoid oversized prompts
        capped = associations[:15]
        summaries = [a["summary"][:200] for a in capped]
        numbered = " ".join(f"({i + 1}) {s}" for i, s in enumerate(summaries))
        user_content = (
            f"These memories surfaced: {numbered}. "
            f"The current context is: {context}. "
            f"What does your gut say?"
        )

        payload = json.dumps({
            "model": VOICE_MODEL,
            "messages": [
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": VOICE_MAX_TOKENS,
            "temperature": VOICE_TEMPERATURE,
        }).encode()

        def _call() -> str | None:
            req = urllib.request.Request(
                f"{VOICE_BASE_URL}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                    content = data["choices"][0]["message"]["content"]
                    return content.strip() if content else None
            except (
                urllib.error.URLError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                OSError,
            ):
                return None

        try:
            return await asyncio.to_thread(_call)
        except Exception:
            return None


# === Unified Orchestrator ===


class InlandEmpire:
    """Subconscious memory layer."""

    def __init__(self) -> None:
        self.config = detect_backends()
        self.graph = GraphBackend(self.config)
        self.semantic = SemanticBackend(self.config)
        self.session = SessionBackend(self.config)
        self.voice = VoiceLayer(self.config.inception_api_key)
        self._initialized = False
        self._init_results: dict[str, Result] = {}

    async def initialize(self) -> None:
        """Initialize all backends."""
        results: dict[str, Result] = {}
        results["graph"] = await self.graph.initialize()
        results["session"] = await self.session.initialize()
        if self.config.mem0_mode:
            results["semantic"] = await self.semantic.initialize()
        self._init_results = results
        self._initialized = True

    def _backend_ok(self, name: str) -> bool:
        """Check if a backend initialized successfully."""
        result = self._init_results.get(name)
        return result is not None and result.is_ok()

    def _semantic_available(self) -> bool:
        """Check if semantic backend is configured and initialized."""
        return self.config.mem0_mode is not None and self._backend_ok("semantic")

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.initialize()

    # --- remember ---

    async def remember(
        self, text: str, memory_type: MemoryType | None = None
    ) -> dict[str, Any]:
        """Store a memory. Auto-classifies type unless overridden."""
        if not text or not text.strip():
            return {
                "status": "error",
                "command": "remember",
                "error": {"message": "Memory text cannot be empty", "code": "EMPTY_INPUT"},
            }

        await self._ensure_init()

        inferred = memory_type or classify_memory(text)

        if inferred == MemoryType.CONTEXT:
            result = await self.session.store(text)
        elif inferred == MemoryType.PATTERN:
            if self._semantic_available():
                result = await self.semantic.store(text)
            else:
                # Fallback: store pattern in graph with entity_type="pattern"
                result = await self.graph.store(text, entity_type="pattern")
        else:
            result = await self.graph.store(text, entity_type="fact")

        if result.is_err():
            return {
                "status": "error",
                "command": "remember",
                "error": {"message": result.error.message, "code": result.error.code},
            }

        response: dict[str, Any] = {
            "status": "ok",
            "command": "remember",
            "result": {"stored": True, "inferred_type": inferred.value},
        }
        if memory_type:
            response["result"]["type_override"] = memory_type.value
        return response

    # --- consult ---

    async def consult(
        self,
        query: str,
        depth: SearchDepth = SearchDepth.SHALLOW,
        memory_type: MemoryType | None = None,
    ) -> dict[str, Any]:
        """Actively search stored memories."""
        if not query or not query.strip():
            return {
                "status": "error",
                "command": "consult",
                "error": {"message": "Query cannot be empty", "code": "EMPTY_INPUT"},
            }

        await self._ensure_init()

        limit = 5 if depth == SearchDepth.SHALLOW else 20
        all_entries: list[MemoryEntry] = []
        partial = False

        async def _query_backend(coro: Any) -> list[MemoryEntry]:
            nonlocal partial
            try:
                result = await asyncio.wait_for(coro, timeout=30.0)
                if result.is_ok():
                    return result.value
                partial = True
            except (asyncio.TimeoutError, Exception):
                partial = True
            return []

        # Query facts from graph
        if memory_type is None or memory_type == MemoryType.FACT:
            if self._backend_ok("graph"):
                all_entries.extend(
                    await _query_backend(
                        self.graph.search(query, limit=limit, entity_type="fact")
                    )
                )

        # Query patterns from semantic (or graph fallback)
        if memory_type is None or memory_type == MemoryType.PATTERN:
            if self._semantic_available():
                all_entries.extend(
                    await _query_backend(self.semantic.search(query, limit=limit))
                )
            elif self._backend_ok("graph"):
                all_entries.extend(
                    await _query_backend(
                        self.graph.search(query, limit=limit, entity_type="pattern")
                    )
                )

        # Query context from session
        if memory_type is None or memory_type == MemoryType.CONTEXT:
            if self._backend_ok("session"):
                all_entries.extend(
                    await _query_backend(self.session.search(query, limit=limit))
                )

        results = [
            {
                "summary": e.summary,
                "type": e.type,
                "score": e.score,
                "observed_at": e.observed_at,
            }
            for e in all_entries
        ]

        return {
            "status": "ok",
            "command": "consult",
            "result": {
                "query": query,
                "depth": depth.value,
                "results": results,
                "partial": partial,
            },
        }

    # --- surface ---

    async def surface(
        self, context: str, *, voice_enabled: bool = True
    ) -> dict[str, Any]:
        """Broad associative retrieval across all backends.

        Unlike consult (which searches for specific terms), surface casts a
        wide net looking for anything tangentially relevant to the current
        context. This is the "gut feeling" command.

        When voice_enabled is True and INCEPTION_API_KEY is set, generates
        an associative voice reading via Mercury diffusion LLM.
        """
        if not context or not context.strip():
            return {
                "status": "error",
                "command": "surface",
                "error": {"message": "Context cannot be empty", "code": "EMPTY_INPUT"},
            }

        await self._ensure_init()

        all_entries: list[MemoryEntry] = []
        partial = False

        async def _query(coro: Any) -> list[MemoryEntry]:
            nonlocal partial
            try:
                result = await asyncio.wait_for(coro, timeout=30.0)
                if result.is_ok():
                    return result.value
                partial = True
            except (asyncio.TimeoutError, Exception):
                partial = True
            return []

        # Extract individual keywords for broad matching.
        # Surface searches each keyword independently, unlike consult which
        # uses the exact query string. This is the "wide net" behavior.
        keywords = [w for w in context.split() if len(w) >= 3]
        if not keywords:
            keywords = [context]

        seen: set[str] = set()

        async def _collect(coro: Any) -> None:
            entries = await _query(coro)
            for entry in entries:
                if entry.summary not in seen:
                    seen.add(entry.summary)
                    all_entries.append(entry)

        # Query all backends with each keyword, generous limits, no type filtering
        for keyword in keywords:
            tasks: list[Any] = []
            if self._backend_ok("graph"):
                tasks.append(_collect(self.graph.search(keyword, limit=20)))
            if self._semantic_available():
                tasks.append(_collect(self.semantic.search(keyword, limit=20)))
            if self._backend_ok("session"):
                tasks.append(_collect(self.session.search(keyword, limit=10)))
            if tasks:
                await asyncio.gather(*tasks)

        # Tag relevance based on score (semantic) or position (others)
        associations = []
        for i, entry in enumerate(all_entries):
            if entry.score is not None:
                if entry.score > 0.7:
                    relevance = "high"
                elif entry.score > 0.4:
                    relevance = "medium"
                else:
                    relevance = "low"
            else:
                # Position-based heuristic for backends without scoring
                if i < 3:
                    relevance = "high"
                elif i < 10:
                    relevance = "medium"
                else:
                    relevance = "low"

            associations.append(
                {
                    "summary": entry.summary,
                    "type": entry.type,
                    "relevance": relevance,
                    "observed_at": entry.observed_at,
                }
            )

        # Voice layer: generate associative reading if enabled and associations exist
        voice: str | None = None
        if voice_enabled and associations:
            voice = await self.voice.generate(associations, context)

        return {
            "status": "ok",
            "command": "surface",
            "result": {
                "context": context,
                "associations": associations,
                "voice": voice,
                "partial": partial,
            },
        }

    # --- forget ---

    async def forget(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        before: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Selectively remove memories.

        When both query and --before are provided, uses AND semantics.
        Note: --before only applies to session/context memories (graph and
        semantic backends lack per-entry timestamps).
        """
        await self._ensure_init()

        # Parse duration if provided
        before_dt: datetime | None = None
        if before:
            duration = parse_duration(before)
            if not duration:
                return {
                    "status": "error",
                    "command": "forget",
                    "error": {
                        "message": f"Invalid duration: {before}. Use format like 7d, 24h, 30m.",
                        "code": "INVALID_DURATION",
                    },
                }
            before_dt = datetime.now(timezone.utc) - duration

        if not query and not before:
            return {
                "status": "error",
                "command": "forget",
                "error": {
                    "message": "Provide a query, --before duration, or both.",
                    "code": "MISSING_CRITERIA",
                },
            }

        if dry_run:
            return await self._forget_dry_run(query, memory_type, before_dt)

        total_deleted = 0
        errors: list[dict[str, Any]] = []

        # Delete from graph (facts and fallback patterns)
        if memory_type is None or memory_type in (MemoryType.FACT, MemoryType.PATTERN):
            if self._backend_ok("graph") and query:
                et = memory_type.value if memory_type else None
                result = await self.graph.delete_matching(query, entity_type=et)
                if result.is_ok():
                    total_deleted += result.value
                elif result.is_err():
                    errors.append({
                        "backend": "graph",
                        "message": result.error.message,
                        "code": result.error.code,
                    })

        # Delete from semantic (patterns)
        if memory_type is None or memory_type == MemoryType.PATTERN:
            if self._semantic_available() and query:
                result = await self.semantic.delete_matching(query)
                if result.is_ok():
                    total_deleted += result.value
                elif result.is_err():
                    errors.append({
                        "backend": "semantic",
                        "message": result.error.message,
                        "code": result.error.code,
                    })

        # Delete from session (context)
        # Note: --before filter only applies to session (graph/semantic lack timestamps)
        if memory_type is None or memory_type == MemoryType.CONTEXT:
            if self._backend_ok("session"):
                result = await self.session.delete_matching(
                    query=query, before=before_dt
                )
                if result.is_ok():
                    total_deleted += result.value
                elif result.is_err():
                    errors.append({
                        "backend": "session",
                        "message": result.error.message,
                        "code": result.error.code,
                    })

        if errors:
            return {
                "status": "error",
                "command": "forget",
                "error": {
                    "message": f"Partial failure: {len(errors)} backend(s) failed",
                    "code": "PARTIAL_FAILURE",
                },
                "result": {"deleted": total_deleted, "errors": errors},
            }

        return {
            "status": "ok",
            "command": "forget",
            "result": {"deleted": total_deleted},
        }

    async def _forget_dry_run(
        self,
        query: str | None,
        memory_type: MemoryType | None,
        before_dt: datetime | None,
    ) -> dict[str, Any]:
        """Preview what would be deleted without actually deleting."""
        would_delete: list[dict[str, Any]] = []

        # Preview graph deletions
        if query and (
            memory_type is None
            or memory_type in (MemoryType.FACT, MemoryType.PATTERN)
        ):
            if self._backend_ok("graph"):
                et = memory_type.value if memory_type else None
                result = await self.graph.search(query, limit=50, entity_type=et)
                if result.is_ok():
                    for e in result.value:
                        would_delete.append({"summary": e.summary, "type": e.type})

        # Preview semantic deletions
        if query and (memory_type is None or memory_type == MemoryType.PATTERN):
            if self._semantic_available():
                result = await self.semantic.search(query, limit=20)
                if result.is_ok():
                    for e in result.value:
                        would_delete.append({"summary": e.summary, "type": e.type})

        # Preview session deletions
        if memory_type is None or memory_type == MemoryType.CONTEXT:
            if self._backend_ok("session"):
                # Get candidates: either matching query or all entries
                if query:
                    result = await self.session.search(query, limit=50)
                else:
                    result = await self.session.all_entries(limit=50)

                if result.is_ok():
                    for e in result.value:
                        # Apply before filter for preview
                        if before_dt and e.observed_at:
                            try:
                                if datetime.fromisoformat(e.observed_at) >= before_dt:
                                    continue
                            except ValueError:
                                continue
                        elif before_dt and not e.observed_at:
                            continue  # No timestamp, skip
                        would_delete.append({"summary": e.summary, "type": e.type})

        return {
            "status": "ok",
            "command": "forget",
            "result": {
                "dry_run": True,
                "would_delete": would_delete,
                "count": len(would_delete),
            },
        }

    # --- stats ---

    async def stats(self) -> dict[str, Any]:
        """Backend health and memory statistics."""
        await self._ensure_init()

        backends: dict[str, Any] = {}

        # Graph
        if self._backend_ok("graph"):
            backends["graph"] = {
                "status": "available",
                "mode": "remote" if self.config.libsql_auth_token else "local",
            }
        else:
            err = self._init_results.get("graph")
            backends["graph"] = {
                "status": "unavailable",
                "error": err.error.message if err and err.is_err() else "unknown",
            }

        # Semantic
        if self._semantic_available():
            backends["semantic"] = {
                "status": "available",
                "mode": self.config.mem0_mode,
            }
        elif self.config.mem0_mode:
            err = self._init_results.get("semantic")
            backends["semantic"] = {
                "status": "unavailable",
                "mode": self.config.mem0_mode,
                "error": err.error.message if err and err.is_err() else "unknown",
            }
        else:
            backends["semantic"] = {
                "status": "disabled",
                "reason": "MEM0_API_KEY not set",
            }

        # Session
        if self._backend_ok("session"):
            count = await self.session.count()
            backends["session"] = {"status": "available", "entries": count}
        else:
            err = self._init_results.get("session")
            backends["session"] = {
                "status": "unavailable",
                "error": err.error.message if err and err.is_err() else "unknown",
            }

        # Voice (Layer 2)
        if self.voice._available:
            backends["voice"] = {
                "status": "available",
                "model": VOICE_MODEL,
                "provider": "inception",
            }
        else:
            backends["voice"] = {
                "status": "disabled",
                "reason": "INCEPTION_API_KEY not set",
            }

        return {
            "status": "ok",
            "command": "stats",
            "result": {"version": VERSION, "backends": backends},
        }

    async def close(self) -> None:
        """Close all backends."""
        await self.graph.close()
        await self.semantic.close()
        await self.session.close()


# === CLI ===


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="inland-empire",
        description="Subconscious memory layer. Absorbs observations, surfaces associative memories as hypotheses.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", required=True)

    # remember
    rem = sub.add_parser("remember", help="Commit something to memory")
    rem.add_argument("text", help="The memory text to store")
    rem.add_argument(
        "--type",
        "-t",
        choices=["fact", "pattern", "context"],
        default=None,
        help="Override auto-classification",
    )

    # consult
    con = sub.add_parser("consult", help="Actively search stored memories")
    con.add_argument("query", help="The query string")
    con.add_argument(
        "--depth",
        "-d",
        choices=["shallow", "deep"],
        default="shallow",
        help="Search depth (default: shallow)",
    )
    con.add_argument(
        "--type",
        "-t",
        choices=["fact", "pattern", "context"],
        default=None,
        help="Filter by memory type",
    )

    # surface
    sur = sub.add_parser("surface", help="Broad associative retrieval")
    sur.add_argument("context", help="Current context to associate against")
    sur.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice layer (skip Mercury diffusion call)",
    )

    # forget
    fgt = sub.add_parser("forget", help="Selectively remove memories")
    fgt.add_argument(
        "query", nargs="?", default=None, help="Query to match for deletion"
    )
    fgt.add_argument(
        "--type",
        "-t",
        choices=["fact", "pattern", "context"],
        default=None,
        help="Restrict to one memory type",
    )
    fgt.add_argument(
        "--before",
        "-b",
        default=None,
        help="Forget session entries older than duration (e.g., 7d, 24h). Only applies to context memories.",
    )
    fgt.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted",
    )

    # stats
    sub.add_parser("stats", help="Backend health and memory statistics")

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Optional config check
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from shared.config_check import require_skill_config

        require_skill_config("inland-empire", output_format="json")
    except ImportError:
        pass  # shared module not available, skip check

    empire = InlandEmpire()

    try:
        match args.command:
            case "remember":
                memory_type = MemoryType(args.type) if args.type else None
                result = await empire.remember(args.text, memory_type)

            case "consult":
                depth = SearchDepth(args.depth)
                memory_type = MemoryType(args.type) if args.type else None
                result = await empire.consult(args.query, depth, memory_type)

            case "surface":
                result = await empire.surface(
                    args.context, voice_enabled=not args.no_voice
                )

            case "forget":
                memory_type = MemoryType(args.type) if args.type else None
                result = await empire.forget(
                    query=args.query,
                    memory_type=memory_type,
                    before=args.before,
                    dry_run=args.dry_run,
                )

            case "stats":
                result = await empire.stats()

            case _:
                result = {
                    "status": "error",
                    "error": {"message": f"Unknown command: {args.command}"},
                }

        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "ok" else 1

    finally:
        await empire.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

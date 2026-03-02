"""Semantic cache for Encyclopedia search results.

Uses embedding similarity to match near-duplicate queries, avoiding redundant
backend calls for paraphrased or repeated questions.

Key design decisions (from SPEC section 3.6):
- Cache by query_type to prevent cross-type contamination (Constitution Rule 4)
- In-memory with JSONL persistence on shutdown
- LRU eviction at 500 entries
- Threshold 0.92 for semantic similarity

Usage:
    from semantic_cache import SemanticCache

    cache = SemanticCache()
    hit = cache.check("kubernetes auth", "library_docs")
    if hit:
        return hit  # cached results

    # ... execute search ...
    cache.store("kubernetes auth", "library_docs", results, ["context7", "exa"])
"""
from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Import-guarded embedding support
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from shared.embeddings import cosine_similarity, encode

    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False

# Import-guarded event emission
try:
    from shared.events import EventType, create_event
    from shared.membrane import get_membrane

    _HAS_EVENTS = True
except ImportError:
    _HAS_EVENTS = False

# Configuration
DEFAULT_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600  # 1 hour
DEFAULT_MAX_ENTRIES = 500
CACHE_DIR = Path.home() / ".encyclopedia" / "cache"
CACHE_FILE = CACHE_DIR / "semantic_cache.jsonl"


@dataclass
class CacheEntry:
    """A cached search result keyed by query embedding."""

    query: str
    query_type: str
    embedding: np.ndarray
    results: list[dict[str, Any]]
    sources_used: list[str]
    created_at: float  # monotonic
    wall_time: float  # time.time() for TTL (survives persistence)
    ttl_seconds: float
    hit_count: int = 0
    last_accessed: float = 0.0  # monotonic, for LRU

    def is_expired(self, now_wall: float | None = None) -> bool:
        wall = now_wall if now_wall is not None else time.time()
        return (wall - self.wall_time) >= self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL persistence."""
        emb_bytes = self.embedding.tobytes()
        return {
            "query": self.query,
            "query_type": self.query_type,
            "embedding_b64": base64.b64encode(emb_bytes).decode("ascii"),
            "embedding_shape": list(self.embedding.shape),
            "embedding_dtype": str(self.embedding.dtype),
            "results": self.results,
            "sources_used": self.sources_used,
            "wall_time": self.wall_time,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry | None:
        """Deserialize from JSONL. Returns None if expired or malformed."""
        try:
            emb_bytes = base64.b64decode(data["embedding_b64"])
            dtype = np.dtype(data.get("embedding_dtype", "float32"))
            shape = tuple(data.get("embedding_shape", [-1]))
            embedding = np.frombuffer(emb_bytes, dtype=dtype).reshape(shape).copy()

            wall_time = data["wall_time"]
            ttl = data.get("ttl_seconds", DEFAULT_TTL_SECONDS)

            now = time.time()
            if (now - wall_time) >= ttl:
                return None  # expired

            mono = time.monotonic()
            return cls(
                query=data["query"],
                query_type=data["query_type"],
                embedding=embedding,
                results=data.get("results", []),
                sources_used=data.get("sources_used", []),
                created_at=mono,
                wall_time=wall_time,
                ttl_seconds=ttl,
                hit_count=data.get("hit_count", 0),
                last_accessed=mono,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Failed to deserialize cache entry: %s", exc)
            return None


class SemanticCache:
    """Embedding-similarity cache for Encyclopedia search results.

    Checks incoming queries against cached query embeddings using cosine
    similarity. Hits above the threshold return cached results immediately.

    Thread safety: Not thread-safe. Designed for single-process CLI use.
    """

    def __init__(
        self,
        threshold: float | None = None,
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        cache_file: Path | None = None,
        auto_persist: bool = True,
    ) -> None:
        self._threshold = threshold or float(
            os.environ.get("ENCYCLOPEDIA_CACHE_THRESHOLD", DEFAULT_THRESHOLD)
        )
        self._ttl = ttl_seconds or float(
            os.environ.get("ENCYCLOPEDIA_CACHE_TTL", DEFAULT_TTL_SECONDS)
        )
        self._max_entries = max_entries or DEFAULT_MAX_ENTRIES
        self._cache_file = cache_file or CACHE_FILE
        self._entries: list[CacheEntry] = []

        if auto_persist:
            atexit.register(self.save)

    @property
    def size(self) -> int:
        return len(self._entries)

    def check(
        self,
        query: str,
        query_type: str,
    ) -> list[dict[str, Any]] | None:
        """Check cache for a semantically similar query.

        Returns cached results on hit, None on miss.
        Only matches within the same query_type (Constitution Rule 4).
        """
        if not _HAS_EMBEDDINGS:
            self._emit_miss(query)
            return None

        query_embedding = encode([query])
        if query_embedding is None:
            self._emit_miss(query)
            return None

        query_vec = query_embedding[0]
        now_wall = time.time()
        now_mono = time.monotonic()

        best_score = 0.0
        best_entry: CacheEntry | None = None

        for entry in self._entries:
            if entry.query_type != query_type:
                continue
            if entry.is_expired(now_wall):
                continue

            similarity = float(cosine_similarity(query_vec, entry.embedding.reshape(1, -1))[0])
            if similarity >= self._threshold and similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_entry is not None:
            best_entry.hit_count += 1
            best_entry.last_accessed = now_mono
            self._emit_hit(query, best_entry.hit_count)
            return best_entry.results

        self._emit_miss(query)
        return None

    def store(
        self,
        query: str,
        query_type: str,
        results: list[dict[str, Any]],
        sources_used: list[str],
    ) -> bool:
        """Store search results in the cache.

        Returns True if stored successfully, False if embeddings unavailable.
        """
        if not _HAS_EMBEDDINGS:
            return False

        query_embedding = encode([query])
        if query_embedding is None:
            return False

        now_mono = time.monotonic()
        entry = CacheEntry(
            query=query,
            query_type=query_type,
            embedding=query_embedding[0],
            results=results,
            sources_used=sources_used,
            created_at=now_mono,
            wall_time=time.time(),
            ttl_seconds=self._ttl,
            last_accessed=now_mono,
        )

        self._entries.append(entry)
        self._evict_if_needed()
        return True

    def _evict_if_needed(self) -> None:
        """Evict expired entries, then LRU if still over max."""
        now_wall = time.time()
        self._entries = [e for e in self._entries if not e.is_expired(now_wall)]

        if len(self._entries) > self._max_entries:
            self._entries.sort(key=lambda e: e.last_accessed)
            self._entries = self._entries[-self._max_entries :]

    def save(self) -> None:
        """Persist cache to JSONL file."""
        if not self._entries:
            return

        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            now_wall = time.time()
            with open(self._cache_file, "w") as f:
                for entry in self._entries:
                    if not entry.is_expired(now_wall):
                        f.write(json.dumps(entry.to_dict()) + "\n")
            logger.debug("Saved %d cache entries to %s", len(self._entries), self._cache_file)
        except OSError as exc:
            logger.warning("Failed to save semantic cache: %s", exc)

    def load(self) -> int:
        """Load cache from JSONL file. Returns number of entries loaded."""
        if not self._cache_file.exists():
            return 0

        loaded = 0
        try:
            with open(self._cache_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = CacheEntry.from_dict(data)
                        if entry is not None:
                            self._entries.append(entry)
                            loaded += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as exc:
            logger.warning("Failed to load semantic cache: %s", exc)

        self._evict_if_needed()
        logger.debug("Loaded %d cache entries from %s", loaded, self._cache_file)
        return loaded

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for verbose output."""
        total_hits = sum(e.hit_count for e in self._entries)
        by_type: dict[str, int] = {}
        for e in self._entries:
            by_type[e.query_type] = by_type.get(e.query_type, 0) + 1

        return {
            "size": len(self._entries),
            "max_entries": self._max_entries,
            "total_hits": total_hits,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
            "entries_by_type": by_type,
        }

    def _emit_hit(self, query: str, hit_count: int) -> None:
        if not _HAS_EVENTS:
            return
        try:
            membrane = get_membrane("encyclopedia")
            if membrane and membrane.can_emit(EventType.ENCYCLOPEDIA_CACHE_HIT):
                create_event(
                    event_type=EventType.ENCYCLOPEDIA_CACHE_HIT,
                    source_skill="encyclopedia",
                    payload={"query": query, "hit_count": hit_count},
                )
        except Exception:
            logger.debug("Failed to emit ENCYCLOPEDIA_CACHE_HIT for query=%r", query, exc_info=True)

    def _emit_miss(self, query: str) -> None:
        if not _HAS_EVENTS:
            return
        try:
            membrane = get_membrane("encyclopedia")
            if membrane and membrane.can_emit(EventType.ENCYCLOPEDIA_CACHE_MISS):
                create_event(
                    event_type=EventType.ENCYCLOPEDIA_CACHE_MISS,
                    source_skill="encyclopedia",
                    payload={"query": query},
                )
        except Exception:
            logger.debug("Failed to emit ENCYCLOPEDIA_CACHE_MISS for query=%r", query, exc_info=True)

"""Tests for semantic cache."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from semantic_cache import CacheEntry, SemanticCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(dim: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def _make_results(n: int = 2) -> list[dict]:
    return [{"title": f"Result {i}", "content": f"Content {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# CacheEntry serialization
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_round_trip(self):
        emb = _make_embedding()
        entry = CacheEntry(
            query="test query",
            query_type="library_docs",
            embedding=emb,
            results=_make_results(),
            sources_used=["exa"],
            created_at=time.monotonic(),
            wall_time=time.time(),
            ttl_seconds=3600,
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored is not None
        assert restored.query == "test query"
        assert restored.query_type == "library_docs"
        assert np.allclose(restored.embedding, emb, atol=1e-6)
        assert len(restored.results) == 2

    def test_expired_entry_returns_none(self):
        emb = _make_embedding()
        d = {
            "query": "old query",
            "query_type": "general_search",
            "embedding_b64": __import__("base64").b64encode(emb.tobytes()).decode(),
            "embedding_shape": list(emb.shape),
            "embedding_dtype": str(emb.dtype),
            "results": [],
            "sources_used": [],
            "wall_time": time.time() - 7200,  # 2 hours ago
            "ttl_seconds": 3600,
        }
        assert CacheEntry.from_dict(d) is None

    def test_malformed_entry_returns_none(self):
        assert CacheEntry.from_dict({"garbage": True}) is None

    def test_is_expired(self):
        entry = CacheEntry(
            query="q",
            query_type="library_docs",
            embedding=_make_embedding(),
            results=[],
            sources_used=[],
            created_at=time.monotonic(),
            wall_time=time.time() - 7200,
            ttl_seconds=3600,
        )
        assert entry.is_expired()

    def test_not_expired(self):
        entry = CacheEntry(
            query="q",
            query_type="library_docs",
            embedding=_make_embedding(),
            results=[],
            sources_used=[],
            created_at=time.monotonic(),
            wall_time=time.time(),
            ttl_seconds=3600,
        )
        assert not entry.is_expired()


# ---------------------------------------------------------------------------
# SemanticCache — similarity matching
# ---------------------------------------------------------------------------


def _mock_encode(texts):
    """Deterministic mock: hash the text to produce a stable embedding."""
    dim = 64
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(dim).astype(np.float32)
        result[i] = vec / (np.linalg.norm(vec) + 1e-9)
    return result


class TestSemanticCacheCheck:
    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    @patch("semantic_cache.cosine_similarity")
    def test_cache_hit(self, mock_cos, mock_enc):
        cache = SemanticCache(threshold=0.9, auto_persist=False)
        emb = _make_embedding(seed=1)
        entry = CacheEntry(
            query="kubernetes auth",
            query_type="library_docs",
            embedding=emb,
            results=_make_results(),
            sources_used=["exa"],
            created_at=time.monotonic(),
            wall_time=time.time(),
            ttl_seconds=3600,
            last_accessed=time.monotonic(),
        )
        cache._entries.append(entry)

        # Simulate high similarity
        mock_cos.return_value = np.array([0.95])
        result = cache.check("k8s authentication", "library_docs")
        assert result is not None
        assert len(result) == 2
        assert entry.hit_count == 1

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    @patch("semantic_cache.cosine_similarity")
    def test_cache_miss_low_similarity(self, mock_cos, mock_enc):
        cache = SemanticCache(threshold=0.92, auto_persist=False)
        emb = _make_embedding(seed=1)
        entry = CacheEntry(
            query="kubernetes auth",
            query_type="library_docs",
            embedding=emb,
            results=_make_results(),
            sources_used=["exa"],
            created_at=time.monotonic(),
            wall_time=time.time(),
            ttl_seconds=3600,
            last_accessed=time.monotonic(),
        )
        cache._entries.append(entry)

        mock_cos.return_value = np.array([0.80])
        result = cache.check("python decorators", "library_docs")
        assert result is None

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    @patch("semantic_cache.cosine_similarity")
    def test_query_type_isolation(self, mock_cos, mock_enc):
        """library_docs cache must not be returned for general_search."""
        cache = SemanticCache(threshold=0.9, auto_persist=False)
        emb = _make_embedding(seed=1)
        entry = CacheEntry(
            query="kubernetes auth",
            query_type="library_docs",
            embedding=emb,
            results=_make_results(),
            sources_used=["exa"],
            created_at=time.monotonic(),
            wall_time=time.time(),
            ttl_seconds=3600,
            last_accessed=time.monotonic(),
        )
        cache._entries.append(entry)

        # Even with high similarity, different query_type → miss
        mock_cos.return_value = np.array([0.99])
        result = cache.check("kubernetes auth", "general_search")
        assert result is None

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    @patch("semantic_cache.cosine_similarity")
    def test_expired_entries_skipped(self, mock_cos, mock_enc):
        cache = SemanticCache(threshold=0.9, auto_persist=False)
        emb = _make_embedding(seed=1)
        entry = CacheEntry(
            query="kubernetes auth",
            query_type="library_docs",
            embedding=emb,
            results=_make_results(),
            sources_used=["exa"],
            created_at=time.monotonic(),
            wall_time=time.time() - 7200,  # expired
            ttl_seconds=3600,
            last_accessed=time.monotonic(),
        )
        cache._entries.append(entry)

        mock_cos.return_value = np.array([0.99])
        result = cache.check("kubernetes auth", "library_docs")
        assert result is None

    @patch("semantic_cache._HAS_EMBEDDINGS", False)
    def test_graceful_fallback_no_embeddings(self):
        cache = SemanticCache(auto_persist=False)
        result = cache.check("test query", "library_docs")
        assert result is None


# ---------------------------------------------------------------------------
# SemanticCache — store and LRU
# ---------------------------------------------------------------------------


class TestSemanticCacheStore:
    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_store_and_retrieve(self, mock_enc):
        cache = SemanticCache(threshold=0.0, auto_persist=False)  # threshold=0 to always hit
        results = _make_results()
        cache.store("test query", "library_docs", results, ["exa"])
        assert cache.size == 1

    @patch("semantic_cache._HAS_EMBEDDINGS", False)
    def test_store_fails_without_embeddings(self):
        cache = SemanticCache(auto_persist=False)
        ok = cache.store("test", "library_docs", [], [])
        assert not ok
        assert cache.size == 0

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_lru_eviction(self, mock_enc):
        cache = SemanticCache(max_entries=3, auto_persist=False)
        for i in range(5):
            cache.store(f"query {i}", "library_docs", _make_results(), ["exa"])

        # Should have evicted oldest 2
        assert cache.size == 3
        # Most recent entries should be kept
        queries = {e.query for e in cache._entries}
        assert "query 4" in queries
        assert "query 3" in queries
        assert "query 2" in queries

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_expired_evicted_on_store(self, mock_enc):
        cache = SemanticCache(max_entries=10, auto_persist=False)
        # Add an expired entry manually
        cache._entries.append(CacheEntry(
            query="old",
            query_type="library_docs",
            embedding=_make_embedding(),
            results=[],
            sources_used=[],
            created_at=time.monotonic(),
            wall_time=time.time() - 7200,
            ttl_seconds=3600,
            last_accessed=time.monotonic(),
        ))
        cache.store("new query", "library_docs", _make_results(), ["exa"])
        assert all(not e.is_expired() for e in cache._entries)


# ---------------------------------------------------------------------------
# SemanticCache — persistence
# ---------------------------------------------------------------------------


class TestSemanticCachePersistence:
    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_save_and_load(self, mock_enc, tmp_path):
        cache_file = tmp_path / "cache.jsonl"

        # Save
        cache1 = SemanticCache(cache_file=cache_file, auto_persist=False)
        cache1.store("query 1", "library_docs", _make_results(1), ["exa"])
        cache1.store("query 2", "general_search", _make_results(2), ["perplexity"])
        cache1.save()

        assert cache_file.exists()
        lines = cache_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # Load into fresh cache
        cache2 = SemanticCache(cache_file=cache_file, auto_persist=False)
        loaded = cache2.load()
        assert loaded == 2
        assert cache2.size == 2

    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_load_skips_expired(self, mock_enc, tmp_path):
        cache_file = tmp_path / "cache.jsonl"
        emb = _make_embedding()

        # Write an expired entry manually
        entry_data = {
            "query": "old",
            "query_type": "library_docs",
            "embedding_b64": __import__("base64").b64encode(emb.tobytes()).decode(),
            "embedding_shape": list(emb.shape),
            "embedding_dtype": str(emb.dtype),
            "results": [],
            "sources_used": [],
            "wall_time": time.time() - 7200,
            "ttl_seconds": 3600,
        }
        cache_file.write_text(json.dumps(entry_data) + "\n")

        cache = SemanticCache(cache_file=cache_file, auto_persist=False)
        loaded = cache.load()
        assert loaded == 0

    def test_load_nonexistent_file(self, tmp_path):
        cache = SemanticCache(cache_file=tmp_path / "nope.jsonl", auto_persist=False)
        loaded = cache.load()
        assert loaded == 0


# ---------------------------------------------------------------------------
# SemanticCache — stats
# ---------------------------------------------------------------------------


class TestSemanticCacheStats:
    @patch("semantic_cache._HAS_EMBEDDINGS", True)
    @patch("semantic_cache.encode", side_effect=_mock_encode)
    def test_stats(self, mock_enc):
        cache = SemanticCache(auto_persist=False)
        cache.store("q1", "library_docs", [], ["exa"])
        cache.store("q2", "general_search", [], ["perplexity"])
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["entries_by_type"]["library_docs"] == 1
        assert stats["entries_by_type"]["general_search"] == 1

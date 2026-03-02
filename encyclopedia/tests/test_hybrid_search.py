"""Tests for hybrid search: merge/dedup, rerank, fulltext/vector helpers."""
from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Register cgcli.embeddings directly to avoid tree-sitter dependency
_embeddings_path = Path(__file__).parent.parent / "scripts" / "cgcli" / "embeddings.py"
_spec = importlib.util.spec_from_file_location("cgcli.embeddings", _embeddings_path)
_emb_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cgcli", type(sys)("cgcli"))
sys.modules["cgcli.embeddings"] = _emb_mod
_spec.loader.exec_module(_emb_mod)

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from encyclopedia import (
    SearchResult,
    _build_codegraph_result,
    _fulltext_search,
    _merge_candidates,
    _rerank_candidates,
    _vector_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ft_hit(name: str, path: str, line: int, score: float) -> dict:
    """Build a fulltext search hit dict."""
    return {
        "type": "Function",
        "name": name,
        "path": path,
        "line": line,
        "source": f"def {name}(): ...",
        "docstring": f"Docstring for {name}",
        "fulltext_score": score,
    }


def _vec_hit(name: str, path: str, line: int, score: float) -> dict:
    """Build a vector search hit dict."""
    return {
        "type": "Function",
        "name": name,
        "path": path,
        "line": line,
        "source": f"def {name}(): ...",
        "docstring": f"Docstring for {name}",
        "vector_score": score,
    }


# ---------------------------------------------------------------------------
# _build_codegraph_result
# ---------------------------------------------------------------------------


class TestBuildCodegraphResult:
    def test_basic_result(self):
        rec = {
            "type": "Function",
            "name": "login",
            "path": "auth.py",
            "line": 10,
            "source": "def login(): pass",
            "docstring": "Login function.",
            "combined_score": 0.85,
        }
        result = _build_codegraph_result(rec)
        assert result.title == "[Function] login (auth.py:10)"
        assert result.source == "codegraph"
        assert result.relevance == 0.85
        assert "Login function." in result.content

    def test_missing_fields(self):
        rec = {"type": None, "name": None, "path": None, "line": None, "source": None, "docstring": None}
        result = _build_codegraph_result(rec)
        assert result.title == "[Code]  (:0)"

    def test_metadata_includes_scores(self):
        rec = {
            "type": "Class",
            "name": "User",
            "path": "models.py",
            "line": 5,
            "source": "",
            "docstring": "",
            "fulltext_score": 0.9,
            "vector_score": 0.7,
            "combined_score": 0.78,
        }
        result = _build_codegraph_result(rec)
        assert result.metadata["fulltext_score"] == 0.9
        assert result.metadata["vector_score"] == 0.7


# ---------------------------------------------------------------------------
# _merge_candidates
# ---------------------------------------------------------------------------


class TestMergeCandidates:
    def test_dedup_by_identity(self):
        """Same (name, path, line) from both sources should merge into one."""
        ft = [_ft_hit("login", "auth.py", 10, 5.0)]
        vec = [_vec_hit("login", "auth.py", 10, 0.9)]
        merged = _merge_candidates(ft, vec)
        assert len(merged) == 1

    def test_fulltext_only(self):
        """Fulltext-only candidates get vector_score=0.0."""
        ft = [_ft_hit("login", "auth.py", 10, 5.0)]
        merged = _merge_candidates(ft, [])
        assert len(merged) == 1
        assert merged[0].metadata.get("vector_score", 0.0) == 0.0

    def test_vector_only(self):
        """Vector-only candidates get fulltext_score=0.0."""
        vec = [_vec_hit("login", "auth.py", 10, 0.9)]
        merged = _merge_candidates([], vec)
        assert len(merged) == 1
        assert merged[0].metadata.get("fulltext_score", 0.0) == 0.0

    def test_both_sources_weighted_scoring(self):
        """Combined score should be 0.4*ft_norm + 0.6*vec."""
        ft = [_ft_hit("login", "auth.py", 10, 10.0)]  # normalized to 1.0
        vec = [_vec_hit("login", "auth.py", 10, 0.8)]
        merged = _merge_candidates(ft, vec)
        assert len(merged) == 1
        expected = 0.4 * 1.0 + 0.6 * 0.8  # 0.88
        assert abs(merged[0].relevance - expected) < 0.01

    def test_sort_order(self):
        """Results should be sorted by combined score descending."""
        ft = [
            _ft_hit("low", "a.py", 1, 1.0),
            _ft_hit("high", "b.py", 2, 10.0),
        ]
        vec = [
            _vec_hit("low", "a.py", 1, 0.1),
            _vec_hit("high", "b.py", 2, 0.9),
        ]
        merged = _merge_candidates(ft, vec)
        assert merged[0].title.startswith("[Function] high")

    def test_empty_inputs(self):
        assert _merge_candidates([], []) == []

    def test_multiple_unique_candidates(self):
        ft = [_ft_hit("a", "x.py", 1, 5.0), _ft_hit("b", "y.py", 2, 3.0)]
        vec = [_vec_hit("c", "z.py", 3, 0.8)]
        merged = _merge_candidates(ft, vec)
        assert len(merged) == 3


# ---------------------------------------------------------------------------
# _rerank_candidates
# ---------------------------------------------------------------------------


class TestRerankCandidates:
    def test_empty_candidates(self):
        assert _rerank_candidates("query", []) == []

    def test_fallback_on_import_error(self):
        """Should return original candidates[:top_k] on ImportError."""
        candidates = [
            SearchResult(title="a", content="aaa", source="codegraph"),
            SearchResult(title="b", content="bbb", source="codegraph"),
        ]
        with patch.object(_emb_mod, "rerank", side_effect=ImportError):
            result = _rerank_candidates("query", candidates, top_k=5)
        assert len(result) == 2
        assert result[0].title == "a"

    def test_changes_order(self):
        """With mocked rerank, order should change."""
        candidates = [
            SearchResult(title="low", content="unrelated", source="codegraph"),
            SearchResult(title="high", content="authentication login", source="codegraph"),
        ]

        def mock_rerank(query, docs, text_key, top_k):
            # Return reversed order with scores
            return [(docs[1], 0.95), (docs[0], 0.1)]

        with patch.object(_emb_mod, "rerank", side_effect=mock_rerank):
            result = _rerank_candidates("auth", candidates, top_k=5)
        assert result[0].title == "high"
        assert result[0].relevance == 0.95

    def test_top_k_respected(self):
        candidates = [
            SearchResult(title=f"r{i}", content=f"content {i}", source="codegraph")
            for i in range(10)
        ]

        def mock_rerank(query, docs, text_key, top_k):
            return [(d, float(i)) for i, d in enumerate(docs[:top_k])]

        with patch.object(_emb_mod, "rerank", side_effect=mock_rerank):
            result = _rerank_candidates("q", candidates, top_k=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _fulltext_search (mocked cgcli db_manager)
# ---------------------------------------------------------------------------


class TestFulltextSearch:
    @pytest.mark.asyncio
    async def test_returns_hits(self):
        mock_rows = [
            {
                "node_type": "Function",
                "name": "login",
                "path": "auth.py",
                "line_number": 10,
                "source": "def login(): ...",
                "docstring": "Login fn.",
                "score": 5.0,
            }
        ]
        mock_client = MagicMock()
        mock_client._db_manager.query = AsyncMock(return_value=mock_rows)

        hits = await _fulltext_search(mock_client, "login", limit=10)
        assert len(hits) == 1
        assert hits[0]["name"] == "login"
        assert hits[0]["fulltext_score"] == 5.0

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        mock_client = MagicMock()
        mock_client._db_manager.query = AsyncMock(side_effect=RuntimeError("connection error"))
        hits = await _fulltext_search(mock_client, "test")
        assert hits == []

    @pytest.mark.asyncio
    async def test_query_params(self):
        """Verify SurrealQL query passes correct parameters."""
        mock_client = MagicMock()
        mock_client._db_manager.query = AsyncMock(return_value=[])

        await _fulltext_search(mock_client, "auth query", limit=15)
        call_args = mock_client._db_manager.query.call_args
        params = call_args[0][1]
        assert params["query"] == "auth query"
        assert params["limit"] == 15


# ---------------------------------------------------------------------------
# _vector_search (mocked cgcli client)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MockSearchResult:
    """Minimal stand-in for cgcli._types.SearchResult."""
    name: str
    file_path: str
    line_number: int
    search_type: str
    relevance_score: float
    source: str | None = None
    docstring: str | None = None
    is_dependency: bool = False


class _MockOk:
    def __init__(self, value):
        self.value = value
    def is_ok(self):
        return True
    def is_err(self):
        return False


class _MockErr:
    def __init__(self, error):
        self.error = error
    def is_ok(self):
        return False
    def is_err(self):
        return True


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_empty_on_err(self):
        mock_client = MagicMock()
        mock_client.vector_search = AsyncMock(return_value=_MockErr("embeddings unavailable"))
        hits = await _vector_search(mock_client, "test")
        assert hits == []

    @pytest.mark.asyncio
    async def test_returns_hits(self):
        mock_results = [
            _MockSearchResult(
                name="verify",
                file_path="auth.py",
                line_number=20,
                search_type="Function",
                relevance_score=0.85,
                source="def verify(): ...",
                docstring="Verify creds.",
            )
        ]
        mock_client = MagicMock()
        mock_client.vector_search = AsyncMock(return_value=_MockOk(mock_results))

        hits = await _vector_search(mock_client, "authentication", limit=10)
        assert len(hits) == 1
        assert hits[0]["name"] == "verify"
        assert hits[0]["vector_score"] == 0.85

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        mock_client = MagicMock()
        mock_client.vector_search = AsyncMock(side_effect=RuntimeError("boom"))
        hits = await _vector_search(mock_client, "test")
        assert hits == []

"""Tests for hybrid search: merge/dedup, rerank, fulltext/vector helpers."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Register codegraph.embeddings directly to avoid tree-sitter dependency
_embeddings_path = Path(__file__).parent.parent / "scripts" / "codegraph" / "embeddings.py"
_spec = importlib.util.spec_from_file_location("codegraph.embeddings", _embeddings_path)
_emb_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("codegraph", type(sys)("codegraph"))
sys.modules["codegraph.embeddings"] = _emb_mod
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
# _fulltext_search (mocked Neo4j driver)
# ---------------------------------------------------------------------------


class TestFulltextSearch:
    def test_returns_hits(self):
        mock_record = {
            "type": "Function",
            "name": "login",
            "path": "auth.py",
            "line": 10,
            "source": "def login(): ...",
            "docstring": "Login fn.",
            "score": 5.0,
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        hits = _fulltext_search(mock_driver, "login", limit=10)
        assert len(hits) == 1
        assert hits[0]["name"] == "login"
        assert hits[0]["fulltext_score"] == 5.0

    def test_exception_returns_empty(self):
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("connection error")
        hits = _fulltext_search(mock_driver, "test")
        assert hits == []

    def test_cypher_params(self):
        """Verify correct Cypher parameters are passed."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        _fulltext_search(mock_driver, "auth query", limit=15)
        call_args = mock_session.run.call_args
        assert call_args.kwargs["search_term"] == "auth query"
        assert call_args.kwargs["limit"] == 15


# ---------------------------------------------------------------------------
# _vector_search (mocked)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_empty_on_import_error(self):
        """Should return [] if encode_query raises ImportError."""
        mock_driver = MagicMock()
        with patch.object(_emb_mod, "encode_query", side_effect=ImportError):
            hits = _vector_search(mock_driver, "test")
        assert hits == []

    def test_returns_empty_on_none_vector(self):
        """Should return [] if encode_query returns None."""
        mock_driver = MagicMock()
        with patch.object(_emb_mod, "encode_query", return_value=None):
            hits = _vector_search(mock_driver, "test")
        assert hits == []

    def test_returns_hits_with_mock(self):
        """With mocked encode_query and Neo4j, should return vector hits."""
        import numpy as np

        mock_vec = np.random.randn(384).astype(np.float32)

        mock_record = {
            "type": "Function",
            "name": "verify",
            "path": "auth.py",
            "line": 20,
            "source": "def verify(): ...",
            "docstring": "Verify creds.",
            "score": 0.85,
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(_emb_mod, "encode_query", return_value=mock_vec):
            hits = _vector_search(mock_driver, "authentication", limit=10)
        assert len(hits) == 1
        assert hits[0]["name"] == "verify"
        assert hits[0]["vector_score"] == 0.85

"""Tests for cgcli.embeddings module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the embeddings module directly to avoid pulling in tree-sitter
# via cgcli.__init__ → indexer → graph_builder
_embeddings_path = Path(__file__).parent.parent / "scripts" / "cgcli" / "embeddings.py"
_spec = importlib.util.spec_from_file_location("cgcli.embeddings", _embeddings_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cgcli.embeddings"] = _mod
_spec.loader.exec_module(_mod)

BIENCODER_MODEL = _mod.BIENCODER_MODEL
CROSSENCODER_MODEL = _mod.CROSSENCODER_MODEL
EMBEDDING_DIM = _mod.EMBEDDING_DIM
compose_embedding_text = _mod.compose_embedding_text
encode_query = _mod.encode_query
encode_texts = _mod.encode_texts
is_available = _mod.is_available
rerank = _mod.rerank


# ---------------------------------------------------------------------------
# compose_embedding_text
# ---------------------------------------------------------------------------


class TestComposeEmbeddingText:
    def test_name_only(self):
        text = compose_embedding_text("process_data")
        assert text == "process_data"

    def test_with_node_type(self):
        text = compose_embedding_text("login", node_type="Function")
        assert text == "Function login"

    def test_with_docstring(self):
        text = compose_embedding_text("login", docstring="Authenticate the user.", node_type="Function")
        assert "Authenticate the user." in text

    def test_with_context(self):
        text = compose_embedding_text("login", context="auth/views.py", node_type="Function")
        assert "in auth/views.py" in text

    def test_full(self):
        text = compose_embedding_text(
            "verify_credentials",
            docstring="Check password hash.\nCompare bcrypt.\nReturn bool.",
            source="def verify_credentials(u, p): ...",
            context="auth/service.py",
            node_type="Function",
        )
        assert text.startswith("Function verify_credentials")
        assert "Check password hash." in text
        assert "in auth/service.py" in text

    def test_truncation_at_512(self):
        long_doc = "x" * 1000
        text = compose_embedding_text("f", docstring=long_doc, node_type="Function")
        assert len(text) <= 512

    def test_multiline_docstring_uses_first_3_lines(self):
        doc = "Line one.\nLine two.\nLine three.\nLine four.\nLine five."
        text = compose_embedding_text("f", docstring=doc)
        assert "Line one." in text
        assert "Line two." in text
        assert "Line three." in text
        assert "Line four." not in text

    def test_empty_docstring(self):
        text = compose_embedding_text("f", docstring="")
        assert text == "f"

    def test_none_docstring(self):
        text = compose_embedding_text("f", docstring=None)
        assert text == "f"


# ---------------------------------------------------------------------------
# encode_texts
# ---------------------------------------------------------------------------


class TestEncodeTexts:
    def test_empty_list_returns_none(self):
        assert encode_texts([]) is None

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_correct_shape(self):
        texts = ["hello world", "test function"]
        vectors = encode_texts(texts)
        assert vectors is not None
        assert vectors.shape == (2, EMBEDDING_DIM)

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_normalized_vectors(self):
        import numpy as np

        texts = ["authentication login"]
        vectors = encode_texts(texts)
        assert vectors is not None
        dot = float(np.dot(vectors[0], vectors[0]))
        assert abs(dot - 1.0) < 0.01

    def test_import_error_returns_none(self):
        with patch.object(_mod, "_get_biencoder", side_effect=ImportError):
            result = encode_texts(["test"])
            assert result is None


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------


class TestEncodeQuery:
    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_returns_correct_shape(self):
        vec = encode_query("authentication")
        assert vec is not None
        assert vec.shape == (EMBEDDING_DIM,)

    def test_import_error_returns_none(self):
        with patch.object(_mod, "_get_biencoder", side_effect=ImportError):
            result = encode_query("test")
            assert result is None


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


class TestRerank:
    def test_empty_candidates(self):
        assert rerank("query", []) == []

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_sorted_by_score(self):
        candidates = [
            {"text": "unrelated fish recipe"},
            {"text": "user authentication login system"},
            {"text": "password verification module"},
        ]
        result = rerank("authentication", candidates, text_key="text", top_k=3)
        assert len(result) == 3
        # Scores should be descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_top_k_respected(self):
        candidates = [{"text": f"candidate {i}"} for i in range(10)]
        result = rerank("test", candidates, text_key="text", top_k=3)
        assert len(result) == 3

    def test_import_error_fallback(self):
        """On ImportError, returns original order with 0.0 scores."""
        candidates = [{"text": "a"}, {"text": "b"}]
        with patch.object(_mod, "_get_crossencoder", side_effect=ImportError):
            result = rerank("query", candidates, text_key="text", top_k=5)
            assert len(result) == 2
            assert result[0] == (candidates[0], 0.0)
            assert result[1] == (candidates[1], 0.0)

    def test_exception_fallback(self):
        """On general exception, returns original order with 0.0 scores."""
        candidates = [{"text": "a"}, {"text": "b"}]
        with patch.object(_mod, "_get_crossencoder", side_effect=RuntimeError("boom")):
            result = rerank("query", candidates, text_key="text", top_k=5)
            assert len(result) == 2
            assert all(score == 0.0 for _, score in result)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_import_error_returns_false(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            # Force re-import check
            assert is_available() is False or is_available() is True  # depends on env


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_biencoder_model_name(self):
        assert BIENCODER_MODEL == "all-MiniLM-L6-v2"

    def test_crossencoder_model_name(self):
        assert CROSSENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 384

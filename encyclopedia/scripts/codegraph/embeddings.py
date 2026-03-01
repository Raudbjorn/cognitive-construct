"""Shared bi-encoder / cross-encoder module for semantic code search.

Provides lazy-loaded sentence-transformer models with graceful fallback
when the ``sentence-transformers`` package is not installed.

Models (cached locally on host):
- Bi-encoder:    all-MiniLM-L6-v2  (384-dim, ~1 text/sec on CPU)
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

BIENCODER_MODEL = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384


def is_available() -> bool:
    """Return True if sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _get_biencoder():
    """Lazy-load and cache the bi-encoder model."""
    from sentence_transformers import SentenceTransformer

    log.info("Loading bi-encoder model: %s", BIENCODER_MODEL)
    return SentenceTransformer(BIENCODER_MODEL)


@lru_cache(maxsize=1)
def _get_crossencoder():
    """Lazy-load and cache the cross-encoder model."""
    from sentence_transformers import CrossEncoder

    log.info("Loading cross-encoder model: %s", CROSSENCODER_MODEL)
    return CrossEncoder(CROSSENCODER_MODEL)


def compose_embedding_text(
    name: str,
    docstring: str | None = None,
    source: str | None = None,
    context: str | None = None,
    node_type: str | None = None,
) -> str:
    """Build the text string to embed from node properties.

    Format: ``"{type} {name}: {signature}. {docstring_first_3_lines}. in {context}"``
    Capped at 512 characters.
    """
    parts: list[str] = []

    prefix = f"{node_type} {name}" if node_type else name
    parts.append(prefix)

    # Use the first line of source as a signature hint
    if source:
        first_line = source.strip().splitlines()[0].strip()
        if first_line:
            parts.append(f": {first_line}")

    if docstring:
        lines = docstring.strip().splitlines()
        first_lines = " ".join(line.strip() for line in lines[:3])
        if first_lines:
            parts.append(f". {first_lines}")

    if context:
        parts.append(f". in {context}")

    text = "".join(parts)
    return text[:512]


def encode_texts(texts: list[str], batch_size: int = 32):
    """Encode a list of texts into normalized embedding vectors.

    Returns:
        numpy.ndarray of shape ``(n, 384)`` with L2-normalized vectors,
        or ``None`` if sentence-transformers is unavailable.
    """
    if not texts:
        return None

    try:
        model = _get_biencoder()
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return vectors
    except ImportError:
        log.warning("sentence-transformers not installed; skipping encode_texts")
        return None
    except Exception:
        log.exception("Failed to encode texts")
        return None


def encode_query(query: str):
    """Encode a single query string into a normalized embedding vector.

    Returns:
        numpy.ndarray of shape ``(384,)`` or ``None`` on failure.
    """
    try:
        model = _get_biencoder()
        vector = model.encode(query, normalize_embeddings=True)
        return vector
    except ImportError:
        log.warning("sentence-transformers not installed; skipping encode_query")
        return None
    except Exception:
        log.exception("Failed to encode query")
        return None


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    text_key: str = "text",
    top_k: int = 10,
) -> list[tuple[dict[str, Any], float]]:
    """Score candidates with the cross-encoder and return sorted results.

    Args:
        query: The search query.
        candidates: List of candidate dicts, each containing a ``text_key`` field.
        text_key: Key in each candidate dict to use as document text.
        top_k: Maximum number of results to return.

    Returns:
        List of ``(candidate, score)`` tuples sorted by score descending.
        Falls back to original order (with score 0.0) on any error.
    """
    if not candidates:
        return []

    try:
        model = _get_crossencoder()
        pairs = [[query, c.get(text_key, "")] for c in candidates]
        scores = model.predict(pairs)

        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    except ImportError:
        log.warning("sentence-transformers not installed; returning original order")
        return [(c, 0.0) for c in candidates[:top_k]]
    except Exception:
        log.exception("Cross-encoder reranking failed; returning original order")
        return [(c, 0.0) for c in candidates[:top_k]]

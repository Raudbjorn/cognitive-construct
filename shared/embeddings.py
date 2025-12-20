"""Lightweight embeddings for context compression.

Uses Model2Vec for fast, small static embeddings. No PyTorch, no ONNX runtime.
~8MB model, microsecond inference.

Usage:
    from shared.embeddings import compress_results, rank_by_relevance

    # Compress API results to most relevant
    compressed = compress_results(raw_results, query, top_k=5)

    # Or just rank without truncating
    ranked = rank_by_relevance(items, query, key=lambda x: x["content"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-8M"
CACHE_DIR = "~/.cache/cognitive-construct/embeddings"

# Truncation
TRUNCATION_SUFFIX = "..."

T = TypeVar("T")


@dataclass
class ScoredItem:
    """Item with relevance score."""

    item: Any
    score: float


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the embedding model (cached singleton)."""
    try:
        from model2vec import StaticModel

        logger.info(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}")
        model = StaticModel.from_pretrained(DEFAULT_EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
        return model
    except ImportError:
        logger.warning("model2vec not installed. Install with: uv add model2vec")
        return None


def encode(texts: list[str]) -> np.ndarray | None:
    """Encode texts to embeddings.

    Parameters
    ----------
    texts : list[str]
        Texts to encode.

    Returns
    -------
    np.ndarray | None
        Embeddings matrix (n_texts, embedding_dim) or None if model unavailable.
    """
    model = _get_model()
    if model is None:
        return None

    return model.encode(texts)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and documents.

    Parameters
    ----------
    query_vec : np.ndarray
        Query embedding (embedding_dim,).
    doc_vecs : np.ndarray
        Document embeddings (n_docs, embedding_dim).

    Returns
    -------
    np.ndarray
        Similarity scores (n_docs,).
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    return np.dot(doc_norms, query_norm)


def rank_by_relevance(
    items: list[T],
    query: str,
    key: Callable[[T], str] | None = None,
) -> list[ScoredItem]:
    """Rank items by semantic relevance to query.

    Parameters
    ----------
    items : list[T]
        Items to rank.
    query : str
        Query to rank against.
    key : Callable[[T], str] | None
        Function to extract text from item. If None, items must be strings.

    Returns
    -------
    list[ScoredItem]
        Items with scores, sorted by relevance (highest first).
    """
    if not items:
        return []

    # Extract text from items
    if key is None:
        texts = [str(item) for item in items]
    else:
        texts = [key(item) for item in items]

    # Encode
    all_texts = [query, *texts]
    embeddings = encode(all_texts)

    if embeddings is None:
        # Fallback: return items in original order with zero scores
        logger.warning("Embedding model unavailable, returning items unranked")
        return [ScoredItem(item=item, score=0.0) for item in items]

    query_vec = embeddings[0]
    doc_vecs = embeddings[1:]

    # Score and rank
    scores = cosine_similarity(query_vec, doc_vecs)
    ranked_indices = np.argsort(scores)[::-1]

    return [
        ScoredItem(item=items[i], score=float(scores[i]))
        for i in ranked_indices
    ]


def compress_results(
    results: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    content_key: str = "content",
    max_chars_per_result: int | None = 2000,
) -> list[dict[str, Any]]:
    """Compress API results to most relevant items.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Raw API results (each should have a content field).
    query : str
        Query to rank relevance against.
    top_k : int
        Maximum number of results to return.
    content_key : str
        Key in result dict containing text content.
    max_chars_per_result : int | None
        Truncate each result's content. None for no truncation.

    Returns
    -------
    list[dict[str, Any]]
        Top-k most relevant results with relevance scores added.
    """
    if not results:
        return []

    # Filter results that have content
    valid_results = [r for r in results if r.get(content_key)]
    if not valid_results:
        return results[:top_k]

    # Rank by relevance
    ranked = rank_by_relevance(
        valid_results,
        query,
        key=lambda r: r.get(content_key, ""),
    )

    # Take top_k and add scores
    compressed = []
    for scored in ranked[:top_k]:
        result = dict(scored.item)  # Copy
        result["_relevance"] = round(scored.score, 4)

        # Truncate content if needed
        if max_chars_per_result and content_key in result:
            content = result[content_key]
            if len(content) > max_chars_per_result:
                result[content_key] = content[:max_chars_per_result] + TRUNCATION_SUFFIX

        compressed.append(result)

    return compressed


def deduplicate_by_similarity(
    items: list[T],
    threshold: float = 0.85,
    key: Callable[[T], str] | None = None,
) -> list[T]:
    """Remove near-duplicate items based on embedding similarity.

    Parameters
    ----------
    items : list[T]
        Items to deduplicate.
    threshold : float
        Similarity threshold above which items are considered duplicates.
    key : Callable[[T], str] | None
        Function to extract text from item.

    Returns
    -------
    list[T]
        Deduplicated items (keeps first occurrence).
    """
    if len(items) <= 1:
        return items

    # Extract texts
    if key is None:
        texts = [str(item) for item in items]
    else:
        texts = [key(item) for item in items]

    embeddings = encode(texts)
    if embeddings is None:
        return items  # Fallback: no deduplication

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = embeddings / norms

    # Find duplicates (O(n^2) but n is small for API results)
    keep = [True] * len(items)
    for i in range(len(items)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(items)):
            if not keep[j]:
                continue
            similarity = np.dot(normalized[i], normalized[j])
            if similarity > threshold:
                keep[j] = False  # Mark as duplicate

    return [item for item, k in zip(items, keep, strict=True) if k]


def summarize_for_context(
    texts: list[str],
    query: str,
    max_total_chars: int = 4000,
) -> str:
    """Create a condensed context string from multiple texts.

    Ranks texts by relevance, deduplicates, and concatenates up to char limit.

    Parameters
    ----------
    texts : list[str]
        Source texts.
    query : str
        Query for relevance ranking.
    max_total_chars : int
        Maximum total characters in output.

    Returns
    -------
    str
        Condensed context string.
    """
    if not texts:
        return ""

    # Deduplicate
    unique_texts = deduplicate_by_similarity(texts, threshold=0.85)

    # Rank by relevance
    ranked = rank_by_relevance(unique_texts, query)

    # Concatenate up to limit
    parts = []
    total_chars = 0
    for scored in ranked:
        text = str(scored.item)
        if total_chars + len(text) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 100:  # Only add if meaningful
                parts.append(text[:remaining] + TRUNCATION_SUFFIX)
            break
        parts.append(text)
        total_chars += len(text) + 2  # Account for separator

    return "\n\n".join(parts)

"""Reciprocal Rank Fusion for combining results from multiple search sources.

Implements the RRF algorithm for merging ranked result lists from different
search backends. Unlike simple priority-based deduplication, RRF naturally
boosts results that appear across multiple sources (cross-source agreement
is a strong relevance signal).

Ported from TTTTRPS's ``search/fusion.rs`` RRFEngine, generalized for
Encyclopedia's multi-source search architecture.

Algorithm:
    RRF(d) = Σ weight_i / (k + rank_i(d))

where k is a damping constant (default 60), rank_i is the 1-indexed rank
of document d in result set i, and weight_i is the set's weight.

Usage:
    from shared.fusion import RRFEngine, RankedItem, FusionStrategy

    engine = RRFEngine.default()

    # Each source produces ranked results
    exa_results = [RankedItem(item=r, rank=i, source="exa") for i, r in enumerate(exa)]
    ppx_results = [RankedItem(item=r, rank=i, source="perplexity") for i, r in enumerate(ppx)]

    # Fuse with per-source weights
    fused = engine.fuse(
        result_sets=[(0.5, exa_results), (0.5, ppx_results)],
        key_fn=lambda r: r.url,  # dedup key
    )

    for result in fused:
        print(f"{result.score:.4f} (from {result.source_count} sources): {result.item}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Hashable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Defaults derived from TTTTRPS vocabulary/fusion_config
_DEFAULT_K: int = 60
_DEFAULT_MIN_SCORE: float = 0.0
_DEFAULT_MAX_RESULTS: int = 20
_DEFAULT_BM25_WEIGHT: float = 0.4
_DEFAULT_VECTOR_WEIGHT: float = 0.6
_DEFAULT_EXACT_MATCH_BOOST: float = 1.5
_DEFAULT_HEADER_MATCH_BOOST: float = 1.2


@dataclass(frozen=True)
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion.

    Attributes:
        k: RRF constant. Higher values flatten rank differences.
            Standard default is 60 (from the original RRF paper).
        min_score: Exclude results below this threshold after fusion.
        max_results: Maximum results to return.
        normalize_scores: Scale final scores to [0, 1] range.
    """

    k: int = _DEFAULT_K
    min_score: float = _DEFAULT_MIN_SCORE
    max_results: int = _DEFAULT_MAX_RESULTS
    normalize_scores: bool = True

    @classmethod
    def lenient(cls) -> RRFConfig:
        """No min_score, more results. Good for exploration."""
        return cls(min_score=0.0, max_results=100, normalize_scores=True)

    @classmethod
    def strict(cls) -> RRFConfig:
        """Higher threshold, fewer results. Good for precision."""
        return cls(min_score=0.15, max_results=10, normalize_scores=True)


class FusionStrategy(Enum):
    """Pre-defined weight strategies for common use cases.

    Each variant returns (keyword_weight, semantic_weight).
    """

    BALANCED = "balanced"
    KEYWORD_HEAVY = "keyword_heavy"
    SEMANTIC_HEAVY = "semantic_heavy"
    VOCABULARY_OPTIMIZED = "vocabulary_optimized"

    @property
    def weights(self) -> tuple[float, float]:
        match self:
            case FusionStrategy.BALANCED:
                return (0.5, 0.5)
            case FusionStrategy.KEYWORD_HEAVY:
                return (0.7, 0.3)
            case FusionStrategy.SEMANTIC_HEAVY:
                return (0.3, 0.7)
            case FusionStrategy.VOCABULARY_OPTIMIZED:
                return (_DEFAULT_BM25_WEIGHT, _DEFAULT_VECTOR_WEIGHT)

    @staticmethod
    def for_query_type(query_type: str) -> FusionStrategy:
        """Select a strategy based on Encyclopedia query classification.

        library_docs → keyword-heavy (exact API/function names matter)
        general_search → balanced
        code_context → semantic-heavy (intent matters more than keywords)
        """
        match query_type:
            case "library_docs":
                return FusionStrategy.KEYWORD_HEAVY
            case "code_context":
                return FusionStrategy.SEMANTIC_HEAVY
            case _:
                return FusionStrategy.BALANCED

    @staticmethod
    def boost_factors() -> tuple[float, float]:
        """Return (exact_match_boost, header_match_boost)."""
        return (_DEFAULT_EXACT_MATCH_BOOST, _DEFAULT_HEADER_MATCH_BOOST)


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class RankedItem(Generic[T]):
    """A search result with ranking info from a specific source."""

    item: T
    rank: int
    """0-indexed rank within the source's result list."""

    source: str
    """Source identifier (e.g. "exa", "perplexity", "context7")."""

    original_score: float | None = None
    """Score from the source's own ranking, if available."""


@dataclass
class FusedResult(Generic[T]):
    """A result after RRF fusion across sources."""

    item: T
    score: float
    """Final RRF score (normalized to [0, 1] if configured)."""

    source_ranks: dict[str, int] = field(default_factory=dict)
    """Rank from each source that contained this result."""

    source_count: int = 0
    """Number of sources that returned this result."""

    @property
    def sources(self) -> list[str]:
        return list(self.source_ranks.keys())


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class RRFEngine:
    """Reciprocal Rank Fusion engine.

    Generic over any item type T. Uses a caller-provided key function
    for deduplication (identifying the same result across sources).
    """

    def __init__(self, config: RRFConfig | None = None) -> None:
        self._config = config or RRFConfig()

    @classmethod
    def default(cls) -> RRFEngine:
        return cls(RRFConfig())

    @classmethod
    def with_k(cls, k: int) -> RRFEngine:
        return cls(RRFConfig(k=k))

    @property
    def config(self) -> RRFConfig:
        return self._config

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        result_sets: list[tuple[float, list[RankedItem[T]]]],
        key_fn: Callable[[T], Any],
    ) -> list[FusedResult[T]]:
        """Fuse multiple ranked result sets using RRF.

        Args:
            result_sets: List of (weight, results) tuples.
                Weight should be in [0, 1] and represents source importance.
            key_fn: Function to extract a hashable dedup key from each item.
                Items with the same key are considered the same result.

        Returns:
            Fused results sorted by RRF score (descending).
        """
        k = float(self._config.k)
        accumulator: dict[Any, FusedResult[T]] = {}

        for weight, items in result_sets:
            for item in items:
                key = key_fn(item.item)
                rrf_contribution = weight / (k + float(item.rank) + 1.0)

                if key in accumulator:
                    entry = accumulator[key]
                    entry.score += rrf_contribution
                    entry.source_ranks[item.source] = item.rank
                    entry.source_count += 1
                else:
                    accumulator[key] = FusedResult(
                        item=item.item,
                        score=rrf_contribution,
                        source_ranks={item.source: item.rank},
                        source_count=1,
                    )

        # Sort by score descending
        results = sorted(accumulator.values(), key=lambda r: r.score, reverse=True)

        # Normalize scores to [0, 1]
        if self._config.normalize_scores and results:
            max_score = results[0].score
            if max_score > 0:
                for r in results:
                    r.score /= max_score

        # Apply min_score filter
        if self._config.min_score > 0:
            results = [r for r in results if r.score >= self._config.min_score]

        # Truncate to max_results
        return results[: self._config.max_results]

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def fuse_sources(
        self,
        source_results: dict[str, list[T]],
        source_weights: dict[str, float],
        key_fn: Callable[[T], Any],
        score_fn: Callable[[T], float] | None = None,
    ) -> list[FusedResult[T]]:
        """Higher-level API: fuse named source results with named weights.

        This is the API Encyclopedia should call directly.

        Args:
            source_results: {source_name: [results]} — results should already
                be in relevance order (best first).
            source_weights: {source_name: weight} — missing sources get 0 weight.
            key_fn: Dedup key extractor.
            score_fn: Optional function to extract the source's own relevance
                score from an item.

        Returns:
            Fused results sorted by RRF score.
        """
        result_sets: list[tuple[float, list[RankedItem[T]]]] = []

        for source, items in source_results.items():
            weight = source_weights.get(source, 0.0)
            if weight <= 0 or not items:
                continue

            ranked = [
                RankedItem(
                    item=item,
                    rank=idx,
                    source=source,
                    original_score=score_fn(item) if score_fn else None,
                )
                for idx, item in enumerate(items)
            ]
            result_sets.append((weight, ranked))

        return self.fuse(result_sets, key_fn)

    def score_at_rank(self, rank: int, weight: float = 1.0) -> float:
        """Compute the RRF score contribution for a given rank.

        Useful for understanding score distributions and setting thresholds.
        """
        return weight / (float(self._config.k) + float(rank) + 1.0)

    def max_theoretical_score(self, weights: list[float]) -> float:
        """Maximum possible score if a result is rank-0 in all sets."""
        return sum(w / (float(self._config.k) + 1.0) for w in weights)


# ------------------------------------------------------------------
# Source weight presets for Encyclopedia
# ------------------------------------------------------------------

_SOURCE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "encyclopedia" / "resources" / "source_config.json"
)

_FALLBACK_WEIGHTS: dict[str, dict[str, float]] = {
    "library_docs": {"context7": 0.60, "exa": 0.40},
    "general_search": {"exa": 0.40, "perplexity": 0.40, "kagi": 0.20},
    "code_context": {"mcp_git_ingest": 0.50, "exa": 0.30, "codegraph": 0.20},
    "repository": {"mcp_git_ingest": 1.0},
}

_DEFAULT_WEIGHTS: dict[str, float] = {"exa": 0.34, "perplexity": 0.33, "context7": 0.33}


def _load_encyclopedia_weights() -> dict[str, dict[str, float]]:
    """Load weights from encyclopedia/resources/source_config.json.

    Falls back to _FALLBACK_WEIGHTS if the config file is missing or malformed.
    """
    try:
        config = json.loads(_SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
        weights = config.get("fusion", {}).get("weights_by_query_type", {})
        if not weights:
            logger.warning("source_config.json missing fusion.weights_by_query_type, using fallback")
            return _FALLBACK_WEIGHTS
        return weights
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to load source_config.json (%s), using fallback weights", exc)
        return _FALLBACK_WEIGHTS


ENCYCLOPEDIA_WEIGHTS: dict[str, dict[str, float]] = _load_encyclopedia_weights()


def get_source_weights(query_type: str) -> dict[str, float]:
    """Get source weights for a given query type.

    Falls back to equal weights for unknown types.
    """
    return ENCYCLOPEDIA_WEIGHTS.get(query_type, _DEFAULT_WEIGHTS)

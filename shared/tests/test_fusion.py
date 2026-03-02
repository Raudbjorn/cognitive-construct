"""Tests for shared.fusion module."""

from __future__ import annotations

import pytest

from shared.fusion import (
    RRFConfig,
    RRFEngine,
    RankedItem,
    FusedResult,
    FusionStrategy,
    ENCYCLOPEDIA_WEIGHTS,
    get_source_weights,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def engine() -> RRFEngine:
    return RRFEngine.default()


@pytest.fixture
def strict_engine() -> RRFEngine:
    return RRFEngine(RRFConfig.strict())


def _item(url: str, title: str = "") -> dict:
    return {"url": url, "title": title or url}


def _ranked(items: list[dict], source: str) -> list[RankedItem[dict]]:
    return [
        RankedItem(item=item, rank=i, source=source)
        for i, item in enumerate(items)
    ]


# ------------------------------------------------------------------
# Basic RRF
# ------------------------------------------------------------------

class TestRRFBasics:
    def test_single_source(self, engine: RRFEngine) -> None:
        items = [_item("a"), _item("b"), _item("c")]
        ranked = _ranked(items, "exa")
        results = engine.fuse([(1.0, ranked)], key_fn=lambda x: x["url"])

        assert len(results) == 3
        # First item should have highest score
        assert results[0].item["url"] == "a"
        assert results[0].score > results[1].score

    def test_cross_source_boost(self, engine: RRFEngine) -> None:
        """Result in both sources should rank higher than either alone."""
        exa_items = _ranked([_item("shared"), _item("exa_only")], "exa")
        ppx_items = _ranked([_item("shared"), _item("ppx_only")], "perplexity")

        results = engine.fuse(
            [(0.5, exa_items), (0.5, ppx_items)],
            key_fn=lambda x: x["url"],
        )

        # "shared" should be first (appears in both)
        assert results[0].item["url"] == "shared"
        assert results[0].source_count == 2
        assert "exa" in results[0].sources
        assert "perplexity" in results[0].sources

    def test_weighted_sources(self, engine: RRFEngine) -> None:
        """Higher-weighted source should dominate when results differ."""
        high = _ranked([_item("high_a"), _item("high_b")], "high")
        low = _ranked([_item("low_a"), _item("low_b")], "low")

        results = engine.fuse(
            [(0.9, high), (0.1, low)],
            key_fn=lambda x: x["url"],
        )

        # Top result should be from the high-weight source
        assert results[0].item["url"] == "high_a"

    def test_empty_input(self, engine: RRFEngine) -> None:
        results = engine.fuse([], key_fn=lambda x: x)
        assert results == []

    def test_empty_result_set(self, engine: RRFEngine) -> None:
        results = engine.fuse([(1.0, [])], key_fn=lambda x: x)
        assert results == []


# ------------------------------------------------------------------
# Score normalization
# ------------------------------------------------------------------

class TestNormalization:
    def test_scores_normalized_to_0_1(self, engine: RRFEngine) -> None:
        items = _ranked([_item("a"), _item("b")], "exa")
        results = engine.fuse([(1.0, items)], key_fn=lambda x: x["url"])

        assert results[0].score == 1.0  # Max is always 1.0
        assert 0 < results[1].score < 1.0

    def test_no_normalization(self) -> None:
        engine = RRFEngine(RRFConfig(normalize_scores=False))
        items = _ranked([_item("a")], "exa")
        results = engine.fuse([(1.0, items)], key_fn=lambda x: x["url"])

        # Raw RRF score: 1.0 / (60 + 0 + 1) = ~0.01639
        assert results[0].score == pytest.approx(1.0 / 61.0, rel=1e-4)


# ------------------------------------------------------------------
# Filtering and limits
# ------------------------------------------------------------------

class TestFiltering:
    def test_max_results(self) -> None:
        engine = RRFEngine(RRFConfig(max_results=2))
        items = _ranked([_item(f"item_{i}") for i in range(10)], "exa")
        results = engine.fuse([(1.0, items)], key_fn=lambda x: x["url"])
        assert len(results) == 2

    def test_min_score_filter(self, strict_engine: RRFEngine) -> None:
        items = _ranked([_item(f"item_{i}") for i in range(50)], "exa")
        results = strict_engine.fuse([(1.0, items)], key_fn=lambda x: x["url"])
        # All surviving results should meet the threshold
        for r in results:
            assert r.score >= strict_engine.config.min_score


# ------------------------------------------------------------------
# Source ranks metadata
# ------------------------------------------------------------------

class TestSourceRanks:
    def test_tracks_ranks(self, engine: RRFEngine) -> None:
        exa = _ranked([_item("a"), _item("b")], "exa")
        ppx = _ranked([_item("b"), _item("a")], "perplexity")

        results = engine.fuse(
            [(0.5, exa), (0.5, ppx)],
            key_fn=lambda x: x["url"],
        )

        for r in results:
            if r.item["url"] == "a":
                assert r.source_ranks["exa"] == 0
                assert r.source_ranks["perplexity"] == 1
            elif r.item["url"] == "b":
                assert r.source_ranks["exa"] == 1
                assert r.source_ranks["perplexity"] == 0


# ------------------------------------------------------------------
# fuse_sources convenience API
# ------------------------------------------------------------------

class TestFuseSources:
    def test_named_sources(self, engine: RRFEngine) -> None:
        results = engine.fuse_sources(
            source_results={
                "exa": [_item("a"), _item("b")],
                "perplexity": [_item("b"), _item("c")],
            },
            source_weights={"exa": 0.6, "perplexity": 0.4},
            key_fn=lambda x: x["url"],
        )

        assert len(results) == 3
        # "b" appears in both, should rank high
        urls = [r.item["url"] for r in results]
        assert urls[0] == "b"

    def test_missing_source_weight_ignored(self, engine: RRFEngine) -> None:
        results = engine.fuse_sources(
            source_results={"exa": [_item("a")], "unknown": [_item("b")]},
            source_weights={"exa": 1.0},  # "unknown" has no weight → 0 → skipped
            key_fn=lambda x: x["url"],
        )
        assert len(results) == 1
        assert results[0].item["url"] == "a"

    def test_zero_weight_source_excluded(self, engine: RRFEngine) -> None:
        results = engine.fuse_sources(
            source_results={"exa": [_item("a")], "kagi": [_item("b")]},
            source_weights={"exa": 1.0, "kagi": 0.0},
            key_fn=lambda x: x["url"],
        )
        assert len(results) == 1

    def test_with_score_fn(self, engine: RRFEngine) -> None:
        items = [{"url": "a", "score": 0.95}, {"url": "b", "score": 0.7}]
        results = engine.fuse_sources(
            source_results={"exa": items},
            source_weights={"exa": 1.0},
            key_fn=lambda x: x["url"],
            score_fn=lambda x: x["score"],
        )
        assert len(results) == 2


# ------------------------------------------------------------------
# FusionStrategy
# ------------------------------------------------------------------

class TestFusionStrategy:
    def test_weights_sum(self) -> None:
        for strategy in FusionStrategy:
            kw, sem = strategy.weights
            assert pytest.approx(kw + sem, abs=0.01) == 1.0

    def test_for_query_type(self) -> None:
        assert FusionStrategy.for_query_type("library_docs") == FusionStrategy.KEYWORD_HEAVY
        assert FusionStrategy.for_query_type("code_context") == FusionStrategy.SEMANTIC_HEAVY
        assert FusionStrategy.for_query_type("general_search") == FusionStrategy.BALANCED
        assert FusionStrategy.for_query_type("unknown") == FusionStrategy.BALANCED


# ------------------------------------------------------------------
# Utility methods
# ------------------------------------------------------------------

class TestUtilities:
    def test_score_at_rank(self, engine: RRFEngine) -> None:
        # rank 0, weight 1.0: 1/(60+0+1) = 1/61
        s = engine.score_at_rank(0, 1.0)
        assert s == pytest.approx(1.0 / 61.0, rel=1e-4)

        # rank 10 should be lower
        s10 = engine.score_at_rank(10, 1.0)
        assert s10 < s

    def test_max_theoretical_score(self, engine: RRFEngine) -> None:
        # Two sources with equal weight
        max_s = engine.max_theoretical_score([0.5, 0.5])
        expected = 2 * 0.5 / 61.0
        assert max_s == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------------
# Encyclopedia weight presets
# ------------------------------------------------------------------

class TestEncyclopediaWeights:
    def test_all_query_types_have_weights(self) -> None:
        for qt in ["library_docs", "general_search", "code_context", "repository"]:
            weights = get_source_weights(qt)
            assert len(weights) > 0
            assert all(v > 0 for v in weights.values())

    def test_weights_approximately_sum_to_one(self) -> None:
        for qt, weights in ENCYCLOPEDIA_WEIGHTS.items():
            total = sum(weights.values())
            assert pytest.approx(total, abs=0.01) == 1.0, f"{qt} weights sum to {total}"

    def test_unknown_type_returns_fallback(self) -> None:
        weights = get_source_weights("nonexistent_type")
        assert len(weights) > 0

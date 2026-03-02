"""Tests for source quality profiling and adaptive weights."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from source_profiler import (
    ADJUSTMENT_PER_WINDOW,
    COLD_START_THRESHOLD,
    MAX_DRIFT,
    WINDOW_SIZE,
    SourceProfiler,
)


# ---------------------------------------------------------------------------
# Rolling window basics
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_record_creates_window(self):
        p = SourceProfiler()
        p.record("exa", "library_docs", result_count=3, latency_ms=200)
        metrics = p.get_metrics("exa", "library_docs")
        assert metrics["sample_count"] == 1
        assert metrics["avg_result_count"] == 3.0

    def test_window_maxlen(self):
        p = SourceProfiler(window_size=5)
        for i in range(10):
            p.record("exa", "library_docs", result_count=i)
        metrics = p.get_metrics("exa", "library_docs")
        assert metrics["sample_count"] == 5
        # Only the last 5 (5,6,7,8,9)
        assert metrics["avg_result_count"] == pytest.approx(7.0)

    def test_empty_metrics(self):
        p = SourceProfiler()
        metrics = p.get_metrics("nonexistent", "library_docs")
        assert metrics["sample_count"] == 0
        assert metrics["avg_result_count"] == 0.0

    def test_per_query_type_isolation(self):
        p = SourceProfiler()
        p.record("exa", "library_docs", result_count=5)
        p.record("exa", "general_search", result_count=10)
        assert p.get_metrics("exa", "library_docs")["avg_result_count"] == 5.0
        assert p.get_metrics("exa", "general_search")["avg_result_count"] == 10.0

    def test_timeout_and_empty_rates(self):
        p = SourceProfiler()
        p.record("exa", "library_docs", timeout=True)
        p.record("exa", "library_docs", empty=True)
        p.record("exa", "library_docs", result_count=3)
        p.record("exa", "library_docs", result_count=2)
        metrics = p.get_metrics("exa", "library_docs")
        assert metrics["timeout_rate"] == pytest.approx(0.25)
        assert metrics["empty_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Weight adjustment
# ---------------------------------------------------------------------------


class TestWeightAdjustment:
    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=True)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.5, "context7": 0.5})
    def test_cold_start_returns_static(self, _mock_weights, _mock_flag):
        """Below COLD_START_THRESHOLD samples → static weights."""
        p = SourceProfiler()
        for _ in range(10):
            p.record("exa", "library_docs", result_count=3, latency_ms=100)
        weights = p.get_adjusted_weights("library_docs")
        assert weights["exa"] == 0.5  # unchanged
        assert weights["context7"] == 0.5

    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=True)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.5})
    def test_good_source_gets_positive_adjustment(self, _mock_weights, _mock_flag):
        """A source with great metrics should get a positive weight boost."""
        p = SourceProfiler()
        for _ in range(COLD_START_THRESHOLD + 10):
            p.record(
                "exa", "library_docs",
                result_count=5, latency_ms=100, fusion_rank=1.0,
            )
        weights = p.get_adjusted_weights("library_docs")
        assert weights["exa"] > 0.5

    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=True)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.5})
    def test_bad_source_gets_negative_adjustment(self, _mock_weights, _mock_flag):
        """A source with terrible metrics should get weight reduction."""
        p = SourceProfiler()
        for _ in range(COLD_START_THRESHOLD + 10):
            p.record(
                "exa", "library_docs",
                result_count=0, latency_ms=2000, fusion_rank=10.0,
                timeout=True, empty=True,
            )
        weights = p.get_adjusted_weights("library_docs")
        assert weights["exa"] < 0.5

    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=True)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.5})
    def test_drift_cap_enforced(self, _mock_weights, _mock_flag):
        """Adjustment must not exceed ±20% of base weight."""
        p = SourceProfiler()
        # All perfect or all terrible — doesn't matter, cap must hold
        for _ in range(COLD_START_THRESHOLD + 10):
            p.record(
                "exa", "library_docs",
                result_count=5, latency_ms=0, fusion_rank=0.0,
            )
        weights = p.get_adjusted_weights("library_docs")
        max_allowed = 0.5 * (1.0 + MAX_DRIFT)
        min_allowed = 0.5 * (1.0 - MAX_DRIFT)
        assert min_allowed <= weights["exa"] <= max_allowed

    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=False)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.5})
    def test_flag_disabled_returns_static(self, _mock_weights, _mock_flag):
        """When feature flag is off, profiler records but returns static weights."""
        p = SourceProfiler()
        for _ in range(100):
            p.record("exa", "library_docs", result_count=5)
        weights = p.get_adjusted_weights("library_docs")
        assert weights["exa"] == 0.5

    @patch("source_profiler._HAS_FUSION", False)
    def test_no_fusion_returns_empty(self):
        p = SourceProfiler()
        weights = p.get_adjusted_weights("library_docs")
        assert weights == {}


# ---------------------------------------------------------------------------
# Feedback integration
# ---------------------------------------------------------------------------


class TestFeedbackIntegration:
    @patch("source_profiler._HAS_FEEDBACK", True)
    @patch("source_profiler.FeedbackCollector")
    def test_feedback_score_integrated(self, mock_collector_cls):
        mock_instance = MagicMock()
        mock_instance.get_source_scores.return_value = {"exa": 0.9}
        mock_collector_cls.get_instance.return_value = mock_instance

        p = SourceProfiler()
        for _ in range(10):
            p.record("exa", "library_docs", result_count=3)
        # The feedback score should influence effectiveness
        eff = p._compute_effectiveness("exa", "library_docs")
        assert eff > 0.5  # boosted by high feedback

    @patch("source_profiler._HAS_FEEDBACK", False)
    def test_no_feedback_graceful(self):
        p = SourceProfiler()
        p.record("exa", "library_docs", result_count=3)
        eff = p._compute_effectiveness("exa", "library_docs")
        assert 0.0 <= eff <= 1.0


# ---------------------------------------------------------------------------
# Profile report
# ---------------------------------------------------------------------------


class TestProfileReport:
    @patch("source_profiler._HAS_FUSION", True)
    @patch("source_profiler._HAS_FLAGS", True)
    @patch("source_profiler.get_flag", return_value=False)
    @patch("source_profiler.get_source_weights", return_value={"exa": 0.4, "context7": 0.6})
    def test_profile_report(self, mock_weights, mock_flag):
        p = SourceProfiler()
        p.record("exa", "library_docs", result_count=3)
        report = p.get_profile("library_docs")
        assert report["query_type"] == "library_docs"
        assert not report["adaptive_enabled"]
        assert "exa" in report["sources"]
        assert "metrics" in report["sources"]["exa"]

"""Source quality profiling with rolling window metrics.

Tracks per-source, per-query-type performance to enable adaptive weight
adjustment. Signals collected: result count, latency, fusion rank, timeouts,
empty results, and downstream feedback scores.

Weight adjustment is conservative: ±5% per window, capped at ±20% drift
from static baseline. Gated by ENCYCLOPEDIA_ADAPTIVE_WEIGHTS feature flag.

Usage:
    from source_profiler import SourceProfiler

    profiler = SourceProfiler()
    profiler.record("exa", "library_docs", result_count=3, latency_ms=280)

    # Get adjusted weights (or static if flag disabled)
    weights = profiler.get_adjusted_weights("library_docs")
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import-guarded dependencies
try:
    from shared.fusion import get_source_weights

    _HAS_FUSION = True
except ImportError:
    _HAS_FUSION = False

try:
    from shared.feedback import FeedbackCollector

    _HAS_FEEDBACK = True
except ImportError:
    _HAS_FEEDBACK = False

try:
    from shared.feature_flags import get_flag

    _HAS_FLAGS = True
except ImportError:
    _HAS_FLAGS = False

# Configuration
WINDOW_SIZE = 200
COLD_START_THRESHOLD = 50
ADJUSTMENT_PER_WINDOW = 0.05  # ±5%
MAX_DRIFT = 0.20  # ±20% cap


@dataclass
class QueryRecord:
    """Single query observation for a source."""

    result_count: int = 0
    latency_ms: float = 0.0
    fusion_rank: float = 0.0  # average rank of this source's results after RRF
    timeout: bool = False
    empty: bool = False


class SourceProfiler:
    """Rolling window quality profiler for Encyclopedia sources.

    Maintains per-(source, query_type) deques of QueryRecords.
    Computes effectiveness scores and weight adjustments.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self._window_size = window_size
        # Key: (source, query_type) → deque of QueryRecord
        self._windows: dict[tuple[str, str], deque[QueryRecord]] = {}

    def record(
        self,
        source: str,
        query_type: str,
        *,
        result_count: int = 0,
        latency_ms: float = 0.0,
        fusion_rank: float = 0.0,
        timeout: bool = False,
        empty: bool = False,
    ) -> None:
        """Record a query observation for a source."""
        key = (source, query_type)
        if key not in self._windows:
            self._windows[key] = deque(maxlen=self._window_size)

        self._windows[key].append(
            QueryRecord(
                result_count=result_count,
                latency_ms=latency_ms,
                fusion_rank=fusion_rank,
                timeout=timeout,
                empty=empty,
            )
        )

    def get_metrics(self, source: str, query_type: str) -> dict[str, float]:
        """Get aggregated metrics for a source/query_type pair."""
        key = (source, query_type)
        window = self._windows.get(key)
        if not window:
            return {
                "avg_result_count": 0.0,
                "avg_latency_ms": 0.0,
                "avg_fusion_rank": 0.0,
                "timeout_rate": 0.0,
                "empty_rate": 0.0,
                "sample_count": 0,
            }

        n = len(window)
        return {
            "avg_result_count": sum(r.result_count for r in window) / n,
            "avg_latency_ms": sum(r.latency_ms for r in window) / n,
            "avg_fusion_rank": sum(r.fusion_rank for r in window) / n,
            "timeout_rate": sum(1 for r in window if r.timeout) / n,
            "empty_rate": sum(1 for r in window if r.empty) / n,
            "sample_count": n,
        }

    def _compute_effectiveness(self, source: str, query_type: str) -> float:
        """Compute an effectiveness score in [0, 1] for a source.

        Higher is better. Combines:
        - Result count (more results = better, capped)
        - Latency (lower = better)
        - Fusion rank (lower = better)
        - Timeout rate (lower = better)
        - Empty rate (lower = better)
        - Feedback score (if available)
        """
        metrics = self.get_metrics(source, query_type)
        if metrics["sample_count"] == 0:
            return 0.5  # neutral

        # Normalize each signal to [0, 1]
        # Result count: 0-5+ results → 0-1
        result_score = min(metrics["avg_result_count"] / 5.0, 1.0)

        # Latency: 0-2000ms → 1-0 (lower is better)
        latency_score = max(1.0 - metrics["avg_latency_ms"] / 2000.0, 0.0)

        # Fusion rank: 0-10 → 1-0 (lower rank is better)
        rank_score = max(1.0 - metrics["avg_fusion_rank"] / 10.0, 0.0)

        # Timeout: 0-1 → 1-0
        timeout_score = 1.0 - metrics["timeout_rate"]

        # Empty: 0-1 → 1-0
        empty_score = 1.0 - metrics["empty_rate"]

        # Weighted combination
        effectiveness = (
            0.30 * result_score
            + 0.15 * latency_score
            + 0.25 * rank_score
            + 0.15 * timeout_score
            + 0.15 * empty_score
        )

        # Integrate feedback if available
        feedback_score = self._get_feedback_score(source)
        if feedback_score is not None:
            effectiveness = 0.7 * effectiveness + 0.3 * feedback_score

        return effectiveness

    def _get_feedback_score(self, source: str) -> float | None:
        """Get feedback effectiveness score for a source, if available."""
        if not _HAS_FEEDBACK:
            return None
        try:
            collector = FeedbackCollector.get_instance()
            scores = collector.get_source_scores()
            return scores.get(source)
        except Exception:
            return None

    def _is_adaptive_enabled(self) -> bool:
        """Check the ENCYCLOPEDIA_ADAPTIVE_WEIGHTS feature flag."""
        if not _HAS_FLAGS:
            return False
        return get_flag("ENCYCLOPEDIA_ADAPTIVE_WEIGHTS", False)

    def get_adjusted_weights(self, query_type: str) -> dict[str, float]:
        """Get source weights, optionally adjusted by profiling data.

        When adaptive weights are disabled (default), returns static baseline.
        When enabled but cold (< COLD_START_THRESHOLD samples), returns static.
        When enabled with sufficient data, adjusts within ±20% of baseline.
        """
        if not _HAS_FUSION:
            return {}

        static_weights = get_source_weights(query_type)

        if not self._is_adaptive_enabled():
            return dict(static_weights)

        # Check cold start: need enough samples for each source
        adjusted = {}
        for source, base_weight in static_weights.items():
            key = (source, query_type)
            window = self._windows.get(key)
            sample_count = len(window) if window else 0

            if sample_count < COLD_START_THRESHOLD:
                adjusted[source] = base_weight
                continue

            effectiveness = self._compute_effectiveness(source, query_type)
            # effectiveness is [0, 1], center is 0.5
            # adjustment = (effectiveness - 0.5) * 2 * ADJUSTMENT_PER_WINDOW → [-0.05, +0.05]
            raw_adjustment = (effectiveness - 0.5) * 2.0 * ADJUSTMENT_PER_WINDOW

            # Clamp total drift
            clamped = max(-MAX_DRIFT, min(MAX_DRIFT, raw_adjustment))
            adjusted[source] = base_weight * (1.0 + clamped)

        return adjusted

    def get_profile(self, query_type: str) -> dict[str, Any]:
        """Get full profiling report for a query type (verbose output)."""
        if not _HAS_FUSION:
            return {}

        static_weights = get_source_weights(query_type)
        adjusted = self.get_adjusted_weights(query_type)

        profiles = {}
        for source in static_weights:
            metrics = self.get_metrics(source, query_type)
            effectiveness = self._compute_effectiveness(source, query_type)
            profiles[source] = {
                "metrics": metrics,
                "effectiveness": round(effectiveness, 4),
                "static_weight": static_weights.get(source, 0.0),
                "adjusted_weight": adjusted.get(source, 0.0),
                "drift": round(
                    adjusted.get(source, 0.0) - static_weights.get(source, 0.0), 4
                ),
            }

        return {
            "query_type": query_type,
            "adaptive_enabled": self._is_adaptive_enabled(),
            "sources": profiles,
        }

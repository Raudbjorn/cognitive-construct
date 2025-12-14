"""
Feedback signal collection for Phase 7 emergence architecture.

This module provides:
- FeedbackSignal: Record of usefulness ratings for events/results
- FeedbackCollector: Aggregation and scoring based on feedback
- Source scoring for adaptive routing

Usage:
    from skills.shared.feedback import FeedbackCollector, record_feedback

    # Record feedback for an event
    record_feedback(
        event_id=some_event.id,
        signal_type="useful",
        context={"task": "code search", "query": "React hooks"}
    )

    # Get source effectiveness scores
    collector = FeedbackCollector.get_instance()
    scores = collector.get_source_scores()
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from .events import Event, EventType, create_event


class SignalType(str, Enum):
    """Types of feedback signals."""
    USEFUL = "useful"              # Result was helpful
    NOT_USEFUL = "not_useful"      # Result was not helpful
    PARTIAL = "partial"            # Result was partially helpful
    TIMEOUT = "timeout"            # Operation timed out before completion


@dataclass
class FeedbackSignal:
    """
    A feedback signal rating an event or result.

    Attributes:
        event_id: The event being rated
        signal_type: Type of feedback (useful, not_useful, partial, timeout)
        context: Additional context about the feedback
        timestamp: When the feedback was recorded
        source: Which skill/source produced the rated result
        weight: Signal weight (default 1.0, higher for explicit feedback)
    """
    event_id: UUID
    signal_type: SignalType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": str(self.event_id),
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "source": self.source,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackSignal":
        """Deserialize from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            signal_type=SignalType(data["signal_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            source=data.get("source"),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class FeedbackSummary:
    """Aggregated feedback statistics."""
    total_signals: int
    useful_count: int
    not_useful_count: int
    partial_count: int
    timeout_count: int
    effectiveness_score: float  # 0.0 to 1.0
    window_start: datetime
    window_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_signals": self.total_signals,
            "useful_count": self.useful_count,
            "not_useful_count": self.not_useful_count,
            "partial_count": self.partial_count,
            "timeout_count": self.timeout_count,
            "effectiveness_score": self.effectiveness_score,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


class FeedbackCollector:
    """
    Collects and aggregates feedback signals for emergent behavior.

    Tracks:
    - Source effectiveness (which sources produce useful results)
    - Pattern effectiveness (which patterns lead to good outcomes)
    - Temporal patterns (time-based effectiveness changes)
    """

    _instance: Optional["FeedbackCollector"] = None

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(
            os.environ.get("FEEDBACK_STORAGE_PATH", Path.home() / ".skills" / "feedback")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._signals_file = self.storage_path / "signals.jsonl"
        self._signals_file.touch(exist_ok=True)
        self._lock = asyncio.Lock()

        # In-memory caches for fast lookups
        self._source_scores: Dict[str, List[float]] = defaultdict(list)
        self._pattern_scores: Dict[str, List[float]] = defaultdict(list)
        self._recent_signals: List[FeedbackSignal] = []

    @classmethod
    def get_instance(cls) -> "FeedbackCollector":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    async def record(self, signal: FeedbackSignal) -> None:
        """Record a feedback signal."""
        # Append to storage
        async with self._lock:
            with open(self._signals_file, "a") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")

        # Update in-memory caches
        self._recent_signals.append(signal)
        self._update_scores(signal)

        # Emit feedback event for downstream subscribers
        await self._emit_feedback_event(signal)

    def _update_scores(self, signal: FeedbackSignal) -> None:
        """Update source and pattern scores based on a signal."""
        # Convert signal type to score
        score_map = {
            SignalType.USEFUL: 1.0,
            SignalType.NOT_USEFUL: 0.0,
            SignalType.PARTIAL: 0.5,
            SignalType.TIMEOUT: 0.25,  # Timeout is slightly better than not_useful
        }
        score = score_map.get(signal.signal_type, 0.5) * signal.weight

        # Update source score
        if signal.source:
            self._source_scores[signal.source].append(score)
            # Keep only last 100 signals per source
            if len(self._source_scores[signal.source]) > 100:
                self._source_scores[signal.source] = self._source_scores[signal.source][-100:]

        # Update pattern score if pattern context is present
        pattern = signal.context.get("pattern")
        if pattern:
            self._pattern_scores[pattern].append(score)
            if len(self._pattern_scores[pattern]) > 100:
                self._pattern_scores[pattern] = self._pattern_scores[pattern][-100:]

    async def _emit_feedback_event(self, signal: FeedbackSignal) -> None:
        """Emit a feedback event for the event store."""
        event_type = EventType.FEEDBACK_USEFUL if signal.signal_type == SignalType.USEFUL else (
            EventType.FEEDBACK_NOT_USEFUL if signal.signal_type == SignalType.NOT_USEFUL else
            EventType.FEEDBACK_PARTIAL if signal.signal_type == SignalType.PARTIAL else
            EventType.FEEDBACK_TIMEOUT
        )

        try:
            # Import here to avoid circular dependency
            from skills.skills.inland_empire.event_store import EventStore

            event = create_event(
                event_type=event_type,
                source_skill="system",
                payload={
                    "rated_event_id": str(signal.event_id),
                    "signal_type": signal.signal_type.value,
                    "source": signal.source,
                    "context": signal.context,
                    "weight": signal.weight,
                },
            )
            store = EventStore()
            await store.append(event)
        except Exception:
            # Feedback events are best-effort
            pass

    def aggregate(
        self,
        window: timedelta = timedelta(hours=24),
        source: Optional[str] = None,
    ) -> FeedbackSummary:
        """
        Aggregate feedback signals over a time window.

        Args:
            window: Time window to aggregate over (default: 24 hours)
            source: Optional filter for a specific source

        Returns:
            FeedbackSummary with counts and effectiveness score
        """
        now = datetime.now(timezone.utc)
        window_start = now - window

        # Filter signals in window
        signals = [
            s for s in self._recent_signals
            if s.timestamp >= window_start
            and (source is None or s.source == source)
        ]

        # Count by type
        useful = sum(1 for s in signals if s.signal_type == SignalType.USEFUL)
        not_useful = sum(1 for s in signals if s.signal_type == SignalType.NOT_USEFUL)
        partial = sum(1 for s in signals if s.signal_type == SignalType.PARTIAL)
        timeout = sum(1 for s in signals if s.signal_type == SignalType.TIMEOUT)
        total = len(signals)

        # Calculate effectiveness score
        if total == 0:
            effectiveness = 0.5  # Neutral when no data
        else:
            # Weighted score: useful=1.0, partial=0.5, timeout=0.25, not_useful=0.0
            weighted_sum = useful * 1.0 + partial * 0.5 + timeout * 0.25
            effectiveness = weighted_sum / total

        return FeedbackSummary(
            total_signals=total,
            useful_count=useful,
            not_useful_count=not_useful,
            partial_count=partial,
            timeout_count=timeout,
            effectiveness_score=effectiveness,
            window_start=window_start,
            window_end=now,
        )

    def get_source_scores(self) -> Dict[str, float]:
        """
        Get effectiveness scores for each source.

        Returns a dict mapping source name to score (0.0 to 1.0).
        Sources with no feedback get a neutral score of 0.5.
        """
        scores = {}
        for source, signal_scores in self._source_scores.items():
            if signal_scores:
                scores[source] = sum(signal_scores) / len(signal_scores)
            else:
                scores[source] = 0.5
        return scores

    def get_pattern_effectiveness(self) -> Dict[str, float]:
        """
        Get effectiveness scores for each pattern.

        Returns a dict mapping pattern name to score (0.0 to 1.0).
        """
        scores = {}
        for pattern, signal_scores in self._pattern_scores.items():
            if signal_scores:
                scores[pattern] = sum(signal_scores) / len(signal_scores)
            else:
                scores[pattern] = 0.5
        return scores

    def get_recommended_sources(
        self,
        query_type: Optional[str] = None,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Get sources recommended based on feedback scores.

        Args:
            query_type: Optional filter for query-type-specific recommendations
            threshold: Minimum score to include a source (default: 0.5)

        Returns:
            List of source names sorted by effectiveness (highest first)
        """
        scores = self.get_source_scores()
        recommended = [
            (source, score) for source, score in scores.items()
            if score >= threshold
        ]
        recommended.sort(key=lambda x: x[1], reverse=True)
        return [source for source, _ in recommended]

    async def load_from_disk(self) -> None:
        """Load signals from persistent storage into memory."""
        if not self._signals_file.exists():
            return

        signals: List[FeedbackSignal] = []
        with open(self._signals_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    signal = FeedbackSignal.from_dict(data)
                    signals.append(signal)
                    self._update_scores(signal)
                except (json.JSONDecodeError, KeyError):
                    continue

        # Keep only recent signals in memory (last 1000)
        self._recent_signals = signals[-1000:]


async def record_feedback(
    event_id: UUID,
    signal_type: str | SignalType,
    context: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    weight: float = 1.0,
) -> None:
    """
    Convenience function to record feedback.

    Args:
        event_id: The event being rated
        signal_type: "useful", "not_useful", "partial", or "timeout"
        context: Additional context about the feedback
        source: Which skill/source produced the rated result
        weight: Signal weight (default 1.0)
    """
    if isinstance(signal_type, str):
        signal_type = SignalType(signal_type)

    signal = FeedbackSignal(
        event_id=event_id,
        signal_type=signal_type,
        context=context or {},
        source=source,
        weight=weight,
    )

    collector = FeedbackCollector.get_instance()
    await collector.record(signal)


def get_source_effectiveness() -> Dict[str, float]:
    """Get effectiveness scores for all sources."""
    collector = FeedbackCollector.get_instance()
    return collector.get_source_scores()


def get_recommended_sources(threshold: float = 0.5) -> List[str]:
    """Get sources with effectiveness above the threshold."""
    collector = FeedbackCollector.get_instance()
    return collector.get_recommended_sources(threshold=threshold)

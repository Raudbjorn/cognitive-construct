"""Source health monitoring with circuit breaker pattern.

Tracks per-source availability using a three-state circuit breaker:
  healthy → circuit_open → degraded (probe) → healthy

Each source is wrapped in a SourceHealth instance that records consecutive
failures and enforces cooldown periods before allowing probe attempts.

Usage:
    from source_health import HealthRegistry

    registry = HealthRegistry()

    if registry.is_available("exa"):
        try:
            results = await query_exa(...)
            registry.record_success("exa", latency_ms=340)
        except Exception:
            registry.record_failure("exa")
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import-guarded event emission
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from shared.events import EventType, create_event
    from shared.membrane import get_membrane

    _HAS_EVENTS = True
except ImportError:
    _HAS_EVENTS = False

# Cross-type fallback map (SPEC 3.5)
FALLBACK_ROUTING: dict[str, list[str]] = {
    "library_docs": ["exa", "perplexity"],
    "general_search": ["exa", "perplexity", "kagi"],
    "code_context": ["exa", "context7"],
    "repository": ["exa"],
}


class HealthState(str, Enum):
    """Circuit breaker states."""

    HEALTHY = "healthy"
    CIRCUIT_OPEN = "circuit_open"
    DEGRADED = "degraded"


@dataclass
class SourceHealthSnapshot:
    """Serializable snapshot of a source's health state."""

    source: str
    status: str
    consecutive_failures: int
    last_failure_at: float | None
    last_success_at: float | None
    circuit_opened_at: float | None
    cooldown_seconds: float
    total_queries: int
    total_failures: int
    avg_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_at": self.last_failure_at,
            "last_success_at": self.last_success_at,
            "circuit_opened_at": self.circuit_opened_at,
            "cooldown_seconds": self.cooldown_seconds,
            "total_queries": self.total_queries,
            "total_failures": self.total_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class SourceHealth:
    """Circuit breaker for a single search backend.

    State machine:
        healthy ──[failure_threshold consecutive failures]──> circuit_open
        circuit_open ──[cooldown expires]──> degraded (probe mode)
        degraded ──[probe succeeds]──> healthy
        degraded ──[probe fails]──> circuit_open (reset cooldown)
    """

    def __init__(
        self,
        source: str,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
    ) -> None:
        self.source = source
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown

        self._consecutive_failures: int = 0
        self._circuit_opened_at: float | None = None
        self._last_failure_at: float | None = None
        self._last_success_at: float | None = None

        # Observability metrics
        self._total_queries: int = 0
        self._total_failures: int = 0
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    @property
    def state(self) -> HealthState:
        """Current circuit breaker state."""
        if self._consecutive_failures < self.failure_threshold:
            return HealthState.HEALTHY
        if self._circuit_opened_at is None:
            return HealthState.CIRCUIT_OPEN
        elapsed = time.monotonic() - self._circuit_opened_at
        if elapsed >= self.cooldown:
            return HealthState.DEGRADED
        return HealthState.CIRCUIT_OPEN

    def is_available(self) -> bool:
        """Whether this source should be queried.

        Returns True when healthy or when in probe mode (degraded).
        """
        state = self.state
        return state in (HealthState.HEALTHY, HealthState.DEGRADED)

    def record_success(self, latency_ms: float = 0.0) -> HealthState | None:
        """Record a successful query. Returns previous state if transition occurred."""
        prev_state = self.state
        self._total_queries += 1
        self._consecutive_failures = 0
        self._circuit_opened_at = None
        self._last_success_at = time.monotonic()

        if latency_ms > 0:
            self._latency_sum += latency_ms
            self._latency_count += 1

        new_state = self.state
        if prev_state != new_state:
            return prev_state
        return None

    def record_failure(self) -> HealthState | None:
        """Record a failed query. Returns previous state if transition occurred."""
        prev_state = self.state
        self._total_queries += 1
        self._total_failures += 1
        self._consecutive_failures += 1
        self._last_failure_at = time.monotonic()

        if self._consecutive_failures >= self.failure_threshold:
            self._circuit_opened_at = time.monotonic()

        new_state = self.state
        if prev_state != new_state:
            return prev_state
        return None

    def get_snapshot(self) -> SourceHealthSnapshot:
        """Get a serializable snapshot of current state."""
        avg_latency = (
            self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0
        )
        return SourceHealthSnapshot(
            source=self.source,
            status=self.state.value,
            consecutive_failures=self._consecutive_failures,
            last_failure_at=self._last_failure_at,
            last_success_at=self._last_success_at,
            circuit_opened_at=self._circuit_opened_at,
            cooldown_seconds=self.cooldown,
            total_queries=self._total_queries,
            total_failures=self._total_failures,
            avg_latency_ms=avg_latency,
        )


class HealthRegistry:
    """Manages per-source circuit breakers with lazy initialization.

    Sources are created on first access with default thresholds.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._sources: dict[str, SourceHealth] = {}

    def _get_or_create(self, source: str) -> SourceHealth:
        if source not in self._sources:
            self._sources[source] = SourceHealth(
                source=source,
                failure_threshold=self._failure_threshold,
                cooldown=self._cooldown,
            )
        return self._sources[source]

    def is_available(self, source: str) -> bool:
        """Check if a source is available (healthy or in probe mode)."""
        return self._get_or_create(source).is_available()

    def record_success(self, source: str, latency_ms: float = 0.0) -> None:
        """Record a successful query for a source."""
        health = self._get_or_create(source)
        prev_state = health.record_success(latency_ms)

        # Emit restoration event on transition from degraded → healthy
        if prev_state in (HealthState.DEGRADED, HealthState.CIRCUIT_OPEN):
            self._emit_restored(source)

    def record_failure(self, source: str) -> None:
        """Record a failed query for a source."""
        health = self._get_or_create(source)
        prev_state = health.record_failure()

        # Emit degradation event on transition to circuit_open
        if prev_state is not None and health.state == HealthState.CIRCUIT_OPEN:
            self._emit_degraded(source)

    def get_status(self, source: str) -> SourceHealthSnapshot:
        """Get the current health snapshot for a source."""
        return self._get_or_create(source).get_snapshot()

    def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get health snapshots for all tracked sources."""
        return {
            source: health.get_snapshot().to_dict()
            for source, health in self._sources.items()
        }

    def get_fallback_sources(
        self,
        query_type: str,
        primary_sources: list[str],
    ) -> list[str]:
        """Get fallback sources when all primaries are circuit-broken.

        Only returns sources that are:
        1. In the fallback map for this query type
        2. Not in the primary source list
        3. Currently available (not circuit-broken themselves)
        """
        fallback_candidates = FALLBACK_ROUTING.get(query_type, [])
        primary_set = set(primary_sources)
        return [
            s
            for s in fallback_candidates
            if s not in primary_set and self.is_available(s)
        ]

    def _emit_degraded(self, source: str) -> None:
        """Emit ENCYCLOPEDIA_SOURCE_DEGRADED event (fire-and-forget)."""
        if not _HAS_EVENTS:
            return
        try:
            membrane = get_membrane("encyclopedia")
            if membrane and membrane.can_emit(EventType.ENCYCLOPEDIA_SOURCE_DEGRADED):
                create_event(
                    event_type=EventType.ENCYCLOPEDIA_SOURCE_DEGRADED,
                    source_skill="encyclopedia",
                    payload={"source": source, "reason": "circuit_open"},
                )
                logger.info("Source %s circuit-broken", source)
        except Exception:
            pass

    def _emit_restored(self, source: str) -> None:
        """Emit ENCYCLOPEDIA_SOURCE_RESTORED event (fire-and-forget)."""
        if not _HAS_EVENTS:
            return
        try:
            membrane = get_membrane("encyclopedia")
            if membrane and membrane.can_emit(EventType.ENCYCLOPEDIA_SOURCE_RESTORED):
                create_event(
                    event_type=EventType.ENCYCLOPEDIA_SOURCE_RESTORED,
                    source_skill="encyclopedia",
                    payload={"source": source},
                )
                logger.info("Source %s restored", source)
        except Exception:
            pass

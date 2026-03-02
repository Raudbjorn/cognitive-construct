"""Tests for source health monitoring and circuit breaker."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from source_health import HealthRegistry, HealthState, SourceHealth


# ---------------------------------------------------------------------------
# SourceHealth circuit breaker
# ---------------------------------------------------------------------------


class TestSourceHealth:
    def test_initial_state_healthy(self):
        h = SourceHealth("exa")
        assert h.state == HealthState.HEALTHY
        assert h.is_available()

    def test_single_failure_stays_healthy(self):
        h = SourceHealth("exa", failure_threshold=3)
        h.record_failure()
        assert h.state == HealthState.HEALTHY
        assert h.is_available()

    def test_two_failures_stays_healthy(self):
        h = SourceHealth("exa", failure_threshold=3)
        h.record_failure()
        h.record_failure()
        assert h.state == HealthState.HEALTHY
        assert h.is_available()

    def test_three_failures_opens_circuit(self):
        h = SourceHealth("exa", failure_threshold=3)
        for _ in range(3):
            h.record_failure()
        assert h.state == HealthState.CIRCUIT_OPEN
        assert not h.is_available()

    def test_circuit_open_blocks_queries(self):
        h = SourceHealth("exa", failure_threshold=3, cooldown=60.0)
        for _ in range(3):
            h.record_failure()
        assert not h.is_available()

    def test_cooldown_transitions_to_degraded(self):
        h = SourceHealth("exa", failure_threshold=3, cooldown=0.01)
        for _ in range(3):
            h.record_failure()
        assert h.state == HealthState.CIRCUIT_OPEN
        time.sleep(0.02)
        assert h.state == HealthState.DEGRADED
        assert h.is_available()

    def test_probe_success_restores_healthy(self):
        h = SourceHealth("exa", failure_threshold=3, cooldown=0.01)
        for _ in range(3):
            h.record_failure()
        time.sleep(0.02)
        assert h.state == HealthState.DEGRADED
        h.record_success(latency_ms=100)
        assert h.state == HealthState.HEALTHY
        assert h.is_available()

    def test_probe_failure_resets_cooldown(self):
        h = SourceHealth("exa", failure_threshold=3, cooldown=0.01)
        for _ in range(3):
            h.record_failure()
        time.sleep(0.02)
        assert h.state == HealthState.DEGRADED
        h.record_failure()
        assert h.state == HealthState.CIRCUIT_OPEN
        assert not h.is_available()

    def test_success_resets_consecutive_failures(self):
        h = SourceHealth("exa", failure_threshold=3)
        h.record_failure()
        h.record_failure()
        h.record_success()
        assert h._consecutive_failures == 0
        h.record_failure()
        assert h.state == HealthState.HEALTHY

    def test_record_success_returns_prev_state_on_transition(self):
        h = SourceHealth("exa", failure_threshold=3, cooldown=0.01)
        for _ in range(3):
            h.record_failure()
        time.sleep(0.02)
        prev = h.record_success()
        assert prev == HealthState.DEGRADED

    def test_record_success_returns_none_no_transition(self):
        h = SourceHealth("exa")
        prev = h.record_success()
        assert prev is None

    def test_record_failure_returns_prev_state_on_transition(self):
        h = SourceHealth("exa", failure_threshold=3)
        h.record_failure()
        h.record_failure()
        prev = h.record_failure()
        assert prev == HealthState.HEALTHY

    def test_latency_tracking(self):
        h = SourceHealth("exa")
        h.record_success(latency_ms=100)
        h.record_success(latency_ms=200)
        h.record_success(latency_ms=300)
        snap = h.get_snapshot()
        assert snap.avg_latency_ms == pytest.approx(200.0)

    def test_total_queries_and_failures(self):
        h = SourceHealth("exa")
        h.record_success()
        h.record_success()
        h.record_failure()
        snap = h.get_snapshot()
        assert snap.total_queries == 3
        assert snap.total_failures == 1


class TestSourceHealthSnapshot:
    def test_snapshot_serialization(self):
        h = SourceHealth("exa")
        h.record_success(latency_ms=150)
        snap = h.get_snapshot()
        d = snap.to_dict()
        assert d["source"] == "exa"
        assert d["status"] == "healthy"
        assert d["total_queries"] == 1
        assert d["avg_latency_ms"] == 150.0

    def test_snapshot_circuit_open(self):
        h = SourceHealth("exa", failure_threshold=2)
        h.record_failure()
        h.record_failure()
        snap = h.get_snapshot()
        assert snap.status == "circuit_open"
        assert snap.consecutive_failures == 2


# ---------------------------------------------------------------------------
# HealthRegistry
# ---------------------------------------------------------------------------


class TestHealthRegistry:
    def test_lazy_initialization(self):
        reg = HealthRegistry()
        assert reg.is_available("exa")
        assert "exa" in reg._sources

    def test_record_success(self):
        reg = HealthRegistry()
        reg.record_success("exa", latency_ms=100)
        snap = reg.get_status("exa")
        assert snap.total_queries == 1
        assert snap.avg_latency_ms == 100.0

    def test_record_failure_circuit_breaks(self):
        reg = HealthRegistry(failure_threshold=2)
        reg.record_failure("exa")
        reg.record_failure("exa")
        assert not reg.is_available("exa")

    def test_multi_source_independence(self):
        reg = HealthRegistry(failure_threshold=2)
        reg.record_failure("exa")
        reg.record_failure("exa")
        assert not reg.is_available("exa")
        assert reg.is_available("perplexity")

    def test_get_all_statuses(self):
        reg = HealthRegistry()
        reg.record_success("exa")
        reg.record_success("perplexity")
        statuses = reg.get_all_statuses()
        assert "exa" in statuses
        assert "perplexity" in statuses

    def test_fallback_sources_excludes_primaries(self):
        reg = HealthRegistry()
        fallbacks = reg.get_fallback_sources("library_docs", ["context7"])
        assert "context7" not in fallbacks
        assert "exa" in fallbacks

    def test_fallback_sources_excludes_circuit_broken(self):
        reg = HealthRegistry(failure_threshold=2)
        reg.record_failure("exa")
        reg.record_failure("exa")
        fallbacks = reg.get_fallback_sources("library_docs", ["context7"])
        assert "exa" not in fallbacks

    def test_fallback_sources_empty_when_all_broken(self):
        reg = HealthRegistry(failure_threshold=2)
        reg.record_failure("exa")
        reg.record_failure("exa")
        reg.record_failure("perplexity")
        reg.record_failure("perplexity")
        fallbacks = reg.get_fallback_sources("library_docs", ["context7"])
        assert fallbacks == []

    def test_fallback_sources_unknown_query_type(self):
        reg = HealthRegistry()
        fallbacks = reg.get_fallback_sources("nonexistent_type", ["context7"])
        assert fallbacks == []


class TestHealthRegistryEvents:
    @patch("source_health._HAS_EVENTS", True)
    @patch("source_health.get_membrane")
    @patch("source_health.create_event")
    def test_degraded_event_emitted(self, mock_create, mock_membrane):
        mock_membrane.return_value.can_emit.return_value = True
        reg = HealthRegistry(failure_threshold=2)
        reg.record_failure("exa")
        reg.record_failure("exa")
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["payload"]["source"] == "exa"

    @patch("source_health._HAS_EVENTS", True)
    @patch("source_health.get_membrane")
    @patch("source_health.create_event")
    def test_restored_event_emitted(self, mock_create, mock_membrane):
        mock_membrane.return_value.can_emit.return_value = True
        reg = HealthRegistry(failure_threshold=2, cooldown=0.01)
        reg.record_failure("exa")
        reg.record_failure("exa")
        mock_create.reset_mock()
        time.sleep(0.02)
        reg.record_success("exa")
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]["payload"]["source"] == "exa"

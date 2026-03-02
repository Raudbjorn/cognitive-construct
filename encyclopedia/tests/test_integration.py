"""Integration tests for Encyclopedia pipeline (Phases 3-6).

Tests the full search pipeline with mocked backends: preprocessing → cache
check → health filtering → dispatch → profiler recording → fusion → cache
store → response. Also validates degradation paths and CLI flags.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure scripts/ and project root are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_search_result(**overrides):
    """Create a minimal SearchResult-compatible dict."""
    from encyclopedia import SearchResult

    defaults = {
        "title": "Test Result",
        "content": "Test content about kubernetes authentication.",
        "url": "https://example.com/test",
        "source": "exa",
        "relevance": 0.8,
        "metadata": {},
    }
    defaults.update(overrides)
    return SearchResult(**defaults)


def _make_creds_mock(*, available_sources=None):
    """Return a Credentials mock that passes validation."""
    creds = MagicMock()
    creds.validate.return_value = (True, "")
    creds.exa_key = "fake-exa"
    creds.context7_key = "fake-c7"
    creds.perplexity_key = "fake-ppx"
    creds.kagi_key = None

    available = available_sources if available_sources is not None else {"context7", "exa", "perplexity"}

    def is_source_available(source, _flags):
        return (source in available, None if source in available else "no_key")

    creds.is_source_available = MagicMock(side_effect=is_source_available)
    return creds


# ---------------------------------------------------------------------------
# Full pipeline: cache miss → dispatch → fusion → cache store
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end search with mocked backends."""

    @pytest.mark.asyncio
    async def test_basic_search_returns_results(self):
        """A simple search should dispatch, collect, and return results."""
        from encyclopedia import execute_search, SearchResult

        fake_results = [
            _fake_search_result(source="exa", title="Exa K8s Auth"),
            _fake_search_result(source="context7", title="Context7 K8s Auth"),
        ]

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[fake_results[0]]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[fake_results[1]]),
        ):
            result = await execute_search("kubernetes auth", limit=5)

        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert result["degraded"] is False

    @pytest.mark.asyncio
    async def test_verbose_includes_diagnostics(self):
        """Verbose mode should include query_analysis, source_routing, etc."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth", verbose=True)

        assert result["status"] == "success"
        assert "verbose" in result
        verbose = result["verbose"]
        assert "query_analysis" in verbose
        assert verbose["query_analysis"]["query_type"] in (
            "library_docs", "general_search", "code_context", "repository"
        )
        assert "source_routing" in verbose
        assert "cache_status" in verbose

    @pytest.mark.asyncio
    async def test_dry_run_no_dispatch(self):
        """Dry-run should return analysis without dispatching queries."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        # These should NOT be called
        exa_mock = AsyncMock(side_effect=AssertionError("should not dispatch"))
        c7_mock = AsyncMock(side_effect=AssertionError("should not dispatch"))

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", exa_mock),
            patch("encyclopedia.query_context7", c7_mock),
        ):
            result = await execute_search("kubernetes auth", dry_run=True)

        assert result["status"] == "dry_run"
        assert "target_sources" in result
        assert "effective_query" in result
        exa_mock.assert_not_called()
        c7_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Semantic cache integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:

    @pytest.mark.asyncio
    async def test_cache_hit_skips_dispatch(self):
        """When cache hits, no backend dispatch should occur."""
        from encyclopedia import execute_search

        cached_data = [{"title": "Cached", "content": "cached content", "source": "exa"}]

        cache_mock = MagicMock()
        cache_mock.check.return_value = cached_data
        cache_mock.get_stats.return_value = {"entries": 1, "hits": 1}

        creds = _make_creds_mock()

        exa_mock = AsyncMock(side_effect=AssertionError("should not dispatch"))

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=cache_mock),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", exa_mock),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        assert result["cached"] is True
        assert result["results"] == cached_data
        exa_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_after_fusion(self):
        """On cache miss, results should be stored in cache after fusion."""
        from encyclopedia import execute_search

        cache_mock = MagicMock()
        cache_mock.check.return_value = None  # miss

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=cache_mock),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        cache_mock.store.assert_called_once()
        call_args = cache_mock.store.call_args
        assert call_args[0][0] == "kubernetes auth"  # query


# ---------------------------------------------------------------------------
# Source health integration
# ---------------------------------------------------------------------------


class TestHealthIntegration:

    @pytest.mark.asyncio
    async def test_circuit_open_skips_source(self):
        """A circuit-broken source should be filtered from dispatch."""
        from encyclopedia import execute_search

        registry_mock = MagicMock()

        def is_available(source):
            return source != "context7"  # context7 is circuit-broken

        registry_mock.is_available = MagicMock(side_effect=is_available)
        registry_mock.get_fallback_sources.return_value = []
        registry_mock.get_all_statuses.return_value = {
            "context7": {"state": "circuit_open"},
            "exa": {"state": "healthy"},
        }

        creds = _make_creds_mock()
        c7_mock = AsyncMock(side_effect=AssertionError("should not dispatch"))

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=registry_mock),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_context7", c7_mock),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        c7_mock.assert_not_called()
        # context7 should appear in degradation.missing
        missing_sources = [m["source"] for m in result["degradation"]["missing"]]
        assert "context7" in missing_sources

    @pytest.mark.asyncio
    async def test_source_failure_records_in_health(self):
        """When a source raises an exception, health registry should record failure."""
        from encyclopedia import execute_search

        registry_mock = MagicMock()
        registry_mock.is_available.return_value = True
        registry_mock.get_all_statuses.return_value = {}

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=registry_mock),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, side_effect=RuntimeError("backend down")),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        # record_failure should have been called for context7
        registry_mock.record_failure.assert_called_with("context7")
        # record_success should have been called for exa
        registry_mock.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_all_sources_circuit_open_uses_fallback(self):
        """When all primary sources are circuit-broken, fallback routing kicks in."""
        from encyclopedia import execute_search

        registry_mock = MagicMock()
        registry_mock.is_available.return_value = False  # all broken
        registry_mock.get_fallback_sources.return_value = ["perplexity"]
        registry_mock.get_all_statuses.return_value = {}

        creds = _make_creds_mock(available_sources={"context7", "exa", "perplexity"})

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=registry_mock),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_perplexity", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="perplexity"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        assert "perplexity" in result["sources_used"]


# ---------------------------------------------------------------------------
# Source profiler integration
# ---------------------------------------------------------------------------


class TestProfilerIntegration:

    @pytest.mark.asyncio
    async def test_profiler_records_on_success(self):
        """Profiler should record metrics after each successful source query."""
        from encyclopedia import execute_search

        profiler_mock = MagicMock()
        profiler_mock.get_adjusted_weights.return_value = {}

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=profiler_mock),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="context7"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        # Profiler should have been called for each source
        assert profiler_mock.record.call_count >= 1
        # Check that record was called with source name and query type
        calls = profiler_mock.record.call_args_list
        recorded_sources = {c[0][0] for c in calls}
        assert recorded_sources & {"exa", "context7"}

    @pytest.mark.asyncio
    async def test_profiler_records_on_failure(self):
        """Profiler should record empty=True when source fails."""
        from encyclopedia import execute_search

        profiler_mock = MagicMock()
        profiler_mock.get_adjusted_weights.return_value = {}

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=profiler_mock),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, side_effect=RuntimeError("fail")),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        # Find the call for context7 — should have empty=True
        for call in profiler_mock.record.call_args_list:
            if call[0][0] == "context7":
                assert call[1].get("empty") is True or call[1].get("result_count") == 0
                break
        else:
            pytest.fail("profiler.record not called for failed source context7")


# ---------------------------------------------------------------------------
# Degradation scenarios
# ---------------------------------------------------------------------------


class TestDegradation:

    @pytest.mark.asyncio
    async def test_no_sources_available_returns_error(self):
        """If all sources are unavailable, return BACKEND_UNAVAILABLE error."""
        from encyclopedia import execute_search, ErrorCode

        creds = _make_creds_mock(available_sources=set())

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "error"
        assert result["code"] == ErrorCode.BACKEND_UNAVAILABLE.value
        assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_all_sources_return_empty(self):
        """If all sources return empty results, status should be NOT_FOUND."""
        from encyclopedia import execute_search, ErrorCode

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "error"
        assert result["code"] == ErrorCode.NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_partial_source_failure_still_returns_results(self):
        """If one source fails but another succeeds, results are returned."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, side_effect=TimeoutError("slow")),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"
        assert result["degraded"] is True
        error_sources = [e["source"] for e in result["degradation"]["errors"]]
        assert "context7" in error_sources

    @pytest.mark.asyncio
    async def test_invalid_credentials_returns_config_error(self):
        """Invalid credentials should return CONFIG_ERROR."""
        from encyclopedia import execute_search, ErrorCode

        creds = MagicMock()
        creds.validate.return_value = (False, "EXA_API_KEY not found")

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "error"
        assert result["code"] == ErrorCode.CONFIG_ERROR.value


# ---------------------------------------------------------------------------
# Graceful degradation when modules are absent
# ---------------------------------------------------------------------------


class TestGracefulDegradation:

    @pytest.mark.asyncio
    async def test_pipeline_works_without_health_module(self):
        """Pipeline should function when source_health is not importable."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._HAS_HEALTH", False),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_pipeline_works_without_cache_module(self):
        """Pipeline should function when semantic_cache is not importable."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._HAS_SEMANTIC_CACHE", False),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia._get_source_profiler", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_pipeline_works_without_profiler_module(self):
        """Pipeline should function when source_profiler is not importable."""
        from encyclopedia import execute_search

        creds = _make_creds_mock()

        with (
            patch("encyclopedia.Credentials.load", return_value=creds),
            patch("encyclopedia.FeatureFlags"),
            patch("encyclopedia._HAS_PROFILER", False),
            patch("encyclopedia._get_semantic_cache", return_value=None),
            patch("encyclopedia._get_health_registry", return_value=None),
            patch("encyclopedia.query_exa", new_callable=AsyncMock, return_value=[
                _fake_search_result(source="exa"),
            ]),
            patch("encyclopedia.query_context7", new_callable=AsyncMock, return_value=[]),
        ):
            result = await execute_search("kubernetes auth")

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Synergy interfaces (Phase 6.5-6.6)
# ---------------------------------------------------------------------------


class TestSynergyInterfaces:

    def test_rhetoric_request_context_callable(self):
        """rhetoric_request_context should be importable and callable."""
        from shared.synergies import rhetoric_request_context

        assert callable(rhetoric_request_context)

    def test_encyclopedia_cache_result_callable(self):
        """encyclopedia_cache_result should be importable and callable."""
        from shared.synergies import encyclopedia_cache_result

        assert callable(encyclopedia_cache_result)

    def test_membrane_encyclopedia_registered(self):
        """Encyclopedia membrane should be registered and have expected emits."""
        try:
            from shared.membrane import get_membrane

            membrane = get_membrane("encyclopedia")
            assert membrane is not None
        except ImportError:
            pytest.skip("shared.membrane not available")

    def test_event_types_defined(self):
        """Encyclopedia event types should be defined in shared.events."""
        try:
            from shared.events import EventType

            # These events should exist for Phase 3-4
            assert hasattr(EventType, "ENCYCLOPEDIA_SOURCE_DEGRADED")
            assert hasattr(EventType, "ENCYCLOPEDIA_SOURCE_RESTORED")
            assert hasattr(EventType, "ENCYCLOPEDIA_CACHE_HIT")
            assert hasattr(EventType, "ENCYCLOPEDIA_CACHE_MISS")
        except ImportError:
            pytest.skip("shared.events not available")

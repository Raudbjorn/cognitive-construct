"""Tests for shared.query_pipeline module."""

from __future__ import annotations

import pytest

from shared.query_pipeline import (
    QueryPipeline,
    PipelineConfig,
    ProcessedQuery,
    _normalize,
    _extract_repo_hint,
    _strip_type_prefix,
    _classify,
)
from shared.vocabulary import Vocabulary


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def pipeline() -> QueryPipeline:
    """Pipeline with default dev vocabulary."""
    return QueryPipeline.default()


@pytest.fixture
def pipeline_no_typo() -> QueryPipeline:
    """Pipeline with typo correction disabled."""
    return QueryPipeline(
        vocabulary=Vocabulary.load_default(),
        config=PipelineConfig(enable_typo_correction=False),
    )


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self) -> None:
        assert _normalize("Hello WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert _normalize("  hello   world  ") == "hello world"

    def test_empty(self) -> None:
        assert _normalize("   ") == ""

    def test_tabs_and_newlines(self) -> None:
        assert _normalize("hello\tworld\nfoo") == "hello world foo"


# ------------------------------------------------------------------
# Repo hint extraction
# ------------------------------------------------------------------

class TestRepoHint:
    def test_extracts_repo(self) -> None:
        hint, cleaned = _extract_repo_hint("repo:owner/name describe auth")
        assert hint == "owner/name"
        assert cleaned == "describe auth"

    def test_repo_in_middle(self) -> None:
        hint, cleaned = _extract_repo_hint("describe repo:foo/bar auth flow")
        assert hint == "foo/bar"
        assert cleaned == "describe auth flow"

    def test_no_repo(self) -> None:
        hint, cleaned = _extract_repo_hint("describe auth flow")
        assert hint is None
        assert cleaned == "describe auth flow"

    def test_repo_only(self) -> None:
        hint, cleaned = _extract_repo_hint("repo:owner/repo")
        assert hint == "owner/repo"
        # When nothing left, returns original
        assert cleaned == "repo:owner/repo"


# ------------------------------------------------------------------
# Type prefix stripping
# ------------------------------------------------------------------

class TestTypePrefix:
    def test_doc_prefix(self) -> None:
        qtype, cleaned = _strip_type_prefix("doc: react hooks")
        assert qtype == "library_docs"
        assert cleaned == "react hooks"

    def test_code_prefix(self) -> None:
        qtype, cleaned = _strip_type_prefix("code: auth middleware")
        assert qtype == "code_context"
        assert cleaned == "auth middleware"

    def test_web_prefix(self) -> None:
        qtype, cleaned = _strip_type_prefix("web: latest rust news")
        assert qtype == "general_search"
        assert cleaned == "latest rust news"

    def test_no_prefix(self) -> None:
        qtype, cleaned = _strip_type_prefix("how to use react hooks")
        assert qtype is None
        assert cleaned == "how to use react hooks"


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

class TestClassify:
    def test_url_is_general(self) -> None:
        assert _classify("https://example.com/page") == "general_search"

    def test_code_patterns(self) -> None:
        assert _classify("def authenticate") == "code_context"
        assert _classify("class UserService") == "code_context"
        assert _classify("import asyncio") == "code_context"
        assert _classify("from fastapi import FastAPI") == "code_context"

    def test_time_keywords(self) -> None:
        assert _classify("latest react updates") == "general_search"
        assert _classify("2025 javascript trends") == "general_search"
        assert _classify("current best practices") == "general_search"

    def test_library_keywords(self) -> None:
        assert _classify("how to use supabase") == "library_docs"
        assert _classify("fastapi documentation") == "library_docs"
        assert _classify("react tutorial") == "library_docs"

    def test_default_is_library_docs(self) -> None:
        assert _classify("react hooks") == "library_docs"


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------

class TestPipeline:
    def test_basic_passthrough(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("react hooks")
        assert result.original == "react hooks"
        assert result.corrected == "react hooks"
        assert result.query_type == "library_docs"
        assert result.corrections == []

    def test_typo_correction(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("kuberntes deploymnet")
        assert result.has_corrections is True
        assert "kubernetes" in result.corrected
        assert "deployment" in result.corrected
        assert result.corrections[0].original == "kuberntes"
        assert result.corrections[0].corrected == "kubernetes"

    def test_abbreviation_expansion(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("api k8s")
        assert result.expanded.was_expanded is True
        # Embedding text should NOT have expansions
        assert result.text_for_embedding == "api k8s"
        # Expanded text should have synonyms
        expanded = result.expanded.expanded_text.lower()
        assert "kubernetes" in expanded

    def test_repo_hint_extraction(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("repo:anthropics/sdk describe auth")
        assert result.repo_hint == "anthropics/sdk"
        assert result.query_type == "code_context"
        assert "repo:" not in result.cleaned_query

    def test_explicit_type_prefix(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("web: trending AI papers")
        assert result.query_type == "general_search"
        assert result.cleaned_query == "trending ai papers"

    def test_empty_query(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("   ")
        assert result.corrected == ""
        assert result.query_type == "general_search"

    def test_typo_disabled(self, pipeline_no_typo: QueryPipeline) -> None:
        result = pipeline_no_typo.process("kuberntes")
        assert result.corrections == []
        assert result.corrected == "kuberntes"

    def test_corrections_summary(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("kuberntes depenency")
        summary = result.corrections_summary
        assert summary is not None
        assert "kuberntes → kubernetes" in summary

    def test_no_corrections_summary(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("kubernetes")
        assert result.corrections_summary is None

    def test_to_dict(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("kuberntes api")
        d = result.to_dict()
        assert "original" in d
        assert "corrected" in d
        assert "text_for_embedding" in d
        assert "query_type" in d
        assert "corrections" in d  # because there's a typo
        assert d["corrections"][0]["from"] == "kuberntes"

    def test_suggestions_for_abbreviations(self, pipeline: QueryPipeline) -> None:
        result = pipeline.process("api ssr")
        # Should suggest what the abbreviations mean
        assert len(result.suggestions) > 0

    def test_mixed_correction_and_expansion(self, pipeline: QueryPipeline) -> None:
        """Typo correction should run before expansion."""
        result = pipeline.process("asyncronous api")
        # "asyncronous" should be corrected first
        assert "asynchronous" in result.corrected
        # Then "api" should be expanded for keyword search
        expanded = result.expanded.expanded_text
        assert "application programming interface" in expanded

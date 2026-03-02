"""Tests for shared.vocabulary module."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.vocabulary import Vocabulary, ExpandedQuery, Correction, levenshtein


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def vocab() -> Vocabulary:
    """A small test vocabulary."""
    v = Vocabulary(max_expansions=5)
    v.add_multi_way(["api", "application programming interface"])
    v.add_multi_way(["k8s", "kubernetes"])
    v.add_multi_way(["db", "database"])
    v.add_multi_way(["ssr", "server side rendering"])
    v.add_one_way("frontend", ["react", "vue", "svelte"])
    return v


@pytest.fixture
def vocab_with_misspellings() -> Vocabulary:
    return Vocabulary(
        common_misspellings={
            "kuberntes": "kubernetes",
            "depenency": "dependency",
            "asyncronous": "asynchronous",
        },
        known_terms={"kubernetes", "dependency", "asynchronous", "react"},
        max_expansions=5,
    )


@pytest.fixture
def default_vocab() -> Vocabulary:
    return Vocabulary.load_default()


# ------------------------------------------------------------------
# Multi-way expansion
# ------------------------------------------------------------------

class TestMultiWayExpansion:
    def test_expands_abbreviation(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("api")
        assert "api" in result
        assert "application programming interface" in result

    def test_expands_reverse(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("kubernetes")
        assert "kubernetes" in result
        assert "k8s" in result

    def test_case_insensitive(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("API")
        assert "api" in result
        assert "application programming interface" in result

    def test_no_expansion_for_unknown(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("foobar")
        assert result == ["foobar"]

    def test_original_is_always_first(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("k8s")
        assert result[0] == "k8s"

    def test_max_expansions_limit(self) -> None:
        v = Vocabulary(max_expansions=2)
        v.add_multi_way(["a", "b", "c", "d", "e"])
        result = v.expand_term("a")
        assert len(result) == 2


# ------------------------------------------------------------------
# One-way expansion
# ------------------------------------------------------------------

class TestOneWayExpansion:
    def test_source_expands(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("frontend")
        assert "frontend" in result
        assert "react" in result
        assert "vue" in result

    def test_target_does_not_expand_back(self, vocab: Vocabulary) -> None:
        result = vocab.expand_term("react")
        assert result == ["react"]

    def test_combined_multi_and_one_way(self) -> None:
        v = Vocabulary(max_expansions=10)
        v.add_multi_way(["auth", "authentication"])
        v.add_one_way("auth", ["login", "session"])
        result = v.expand_term("auth")
        assert "auth" in result
        assert "authentication" in result
        assert "login" in result
        assert "session" in result


# ------------------------------------------------------------------
# Query expansion
# ------------------------------------------------------------------

class TestQueryExpansion:
    def test_expand_query(self, vocab: Vocabulary) -> None:
        eq = vocab.expand_query("api k8s")
        assert len(eq.term_groups) == 2
        assert "application programming interface" in eq.term_groups[0]
        assert "kubernetes" in eq.term_groups[1]

    def test_text_for_embedding(self, vocab: Vocabulary) -> None:
        eq = vocab.expand_query("api k8s deploy")
        # Should be just the first (original) term from each group
        assert eq.text_for_embedding == "api k8s deploy"

    def test_expanded_text(self, vocab: Vocabulary) -> None:
        eq = vocab.expand_query("db query")
        text = eq.expanded_text
        assert "db" in text
        assert "database" in text
        assert "query" in text

    def test_was_expanded_true(self, vocab: Vocabulary) -> None:
        eq = vocab.expand_query("api test")
        assert eq.was_expanded is True

    def test_was_expanded_false(self, vocab: Vocabulary) -> None:
        eq = vocab.expand_query("hello world")
        assert eq.was_expanded is False


# ------------------------------------------------------------------
# Typo correction
# ------------------------------------------------------------------

class TestTypoCorrection:
    def test_static_correction(self, vocab_with_misspellings: Vocabulary) -> None:
        assert vocab_with_misspellings.correct_typo("kuberntes") == "kubernetes"
        assert vocab_with_misspellings.correct_typo("depenency") == "dependency"

    def test_known_term_returns_none(self, vocab_with_misspellings: Vocabulary) -> None:
        assert vocab_with_misspellings.correct_typo("kubernetes") is None
        assert vocab_with_misspellings.correct_typo("react") is None

    def test_unknown_no_close_match(self, vocab_with_misspellings: Vocabulary) -> None:
        # "xyzzy" is far from any known term
        assert vocab_with_misspellings.correct_typo("xyzzy") is None

    def test_edit_distance_fallback(self) -> None:
        v = Vocabulary(known_terms={"typescript", "javascript"})
        # One edit away from "typescript"
        assert v.correct_typo("typescrip") == "typescript"


# ------------------------------------------------------------------
# Stop words and known terms
# ------------------------------------------------------------------

class TestFilterAndLookup:
    def test_is_stop_word(self) -> None:
        v = Vocabulary(stop_words={"the", "a", "is"})
        assert v.is_stop_word("the") is True
        assert v.is_stop_word("react") is False

    def test_filter_stop_words(self) -> None:
        v = Vocabulary(stop_words={"the", "a", "is"})
        assert v.filter_stop_words(["the", "api", "is", "broken"]) == ["api", "broken"]

    def test_is_known_term(self, vocab: Vocabulary) -> None:
        assert vocab.is_known_term("api") is True
        assert vocab.is_known_term("kubernetes") is True
        assert vocab.is_known_term("react") is True  # from one-way targets
        assert vocab.is_known_term("xyzzy") is False


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

class TestLoading:
    def test_load_default(self, default_vocab: Vocabulary) -> None:
        stats = default_vocab.stats
        assert stats["multi_way_groups"] > 50
        assert stats["known_terms"] > 100
        assert stats["misspellings"] > 20

    def test_default_expands_k8s(self, default_vocab: Vocabulary) -> None:
        result = default_vocab.expand_term("k8s")
        assert "kubernetes" in result

    def test_default_corrects_kuberntes(self, default_vocab: Vocabulary) -> None:
        assert default_vocab.correct_typo("kuberntes") == "kubernetes"

    def test_load_missing_file(self) -> None:
        v = Vocabulary.load_from_json(Path("/nonexistent/vocab.json"))
        assert v.stats["multi_way_groups"] == 0

    def test_merge_group_on_overlap(self) -> None:
        v = Vocabulary(max_expansions=10)
        v.add_multi_way(["a", "b"])
        v.add_multi_way(["b", "c"])
        # Should merge into one group
        result = v.expand_term("a")
        assert "c" in result

    def test_merge_multiple_groups_on_overlap(self) -> None:
        v = Vocabulary(max_expansions=10)
        v.add_multi_way(["a", "b"])
        v.add_multi_way(["c", "d"])
        # Bridge both existing groups
        v.add_multi_way(["b", "c"])
        result = v.expand_term("a")
        assert {"a", "b", "c", "d"} <= set(result)


# ------------------------------------------------------------------
# Levenshtein
# ------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein("hello", "hello") == 0

    def test_one_insertion(self) -> None:
        assert levenshtein("hello", "helloo") == 1

    def test_one_deletion(self) -> None:
        assert levenshtein("hello", "hell") == 1

    def test_one_substitution(self) -> None:
        assert levenshtein("hello", "hallo") == 1

    def test_empty_strings(self) -> None:
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "abc") == 3

    def test_completely_different(self) -> None:
        assert levenshtein("abc", "xyz") == 3

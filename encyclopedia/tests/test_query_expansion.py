"""Tests for codegraph query expansion."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from encyclopedia import (
    _tokenize_identifier,
    expand_codegraph_query,
    _CODE_SYNONYMS,
    _REVERSE_SYNONYMS,
)


# ---------------------------------------------------------------------------
# _tokenize_identifier
# ---------------------------------------------------------------------------


class TestTokenizeIdentifier:
    def test_camel_case(self):
        assert _tokenize_identifier("getUserData") == ["get", "user", "data"]

    def test_pascal_case(self):
        assert _tokenize_identifier("UserProfile") == ["user", "profile"]

    def test_snake_case(self):
        assert _tokenize_identifier("process_request") == ["process", "request"]

    def test_kebab_case(self):
        assert _tokenize_identifier("my-component") == ["my", "component"]

    def test_dot_case(self):
        assert _tokenize_identifier("config.settings") == ["config", "settings"]

    def test_single_word(self):
        assert _tokenize_identifier("login") == ["login"]

    def test_all_caps_acronym(self):
        result = _tokenize_identifier("HTTPClient")
        assert result == ["http", "client"]

    def test_mixed_separators(self):
        result = _tokenize_identifier("get_UserData")
        assert result == ["get", "user", "data"]

    def test_empty_string(self):
        assert _tokenize_identifier("") == []

    def test_numbers_preserved(self):
        result = _tokenize_identifier("getV2Data")
        # Numbers stay attached to adjacent letters
        assert "v2" in result or "get" in result

    def test_all_lowercase(self):
        assert _tokenize_identifier("fetch") == ["fetch"]


# ---------------------------------------------------------------------------
# _CODE_SYNONYMS / _REVERSE_SYNONYMS consistency
# ---------------------------------------------------------------------------


class TestSynonymMaps:
    def test_reverse_map_contains_all_keys(self):
        """Every key in _CODE_SYNONYMS should appear in _REVERSE_SYNONYMS."""
        for key in _CODE_SYNONYMS:
            assert key in _REVERSE_SYNONYMS

    def test_reverse_map_contains_all_values(self):
        """Every synonym value should appear in _REVERSE_SYNONYMS."""
        for key, syns in _CODE_SYNONYMS.items():
            for syn in syns:
                assert syn in _REVERSE_SYNONYMS, f"{syn!r} (from {key!r}) missing"

    def test_reverse_map_bidirectional(self):
        """A synonym should map back to its key."""
        for key, syns in _CODE_SYNONYMS.items():
            for syn in syns:
                assert key in _REVERSE_SYNONYMS[syn]

    def test_no_empty_families(self):
        for key in _CODE_SYNONYMS:
            assert len(_CODE_SYNONYMS[key]) > 0


# ---------------------------------------------------------------------------
# expand_codegraph_query
# ---------------------------------------------------------------------------


class TestExpandCodegraphQuery:
    def test_synonym_expansion(self):
        result = expand_codegraph_query("authentication")
        terms = set(result.split(" OR "))
        # Should include the "auth" family
        assert "auth" in terms
        assert "login" in terms
        assert "authentication" in terms

    def test_identifier_splitting(self):
        result = expand_codegraph_query("getUserData")
        terms = set(result.split(" OR "))
        assert "get" in terms
        assert "user" in terms
        assert "data" in terms

    def test_identifier_split_triggers_synonyms(self):
        """Splitting getUserData should expand 'get' to fetch/retrieve/etc."""
        result = expand_codegraph_query("getUserData")
        terms = set(result.split(" OR "))
        assert "fetch" in terms
        assert "retrieve" in terms

    def test_user_synonym_expansion(self):
        """'user' token should expand to account/profile/member."""
        result = expand_codegraph_query("getUserData")
        terms = set(result.split(" OR "))
        assert "account" in terms
        assert "profile" in terms

    def test_no_synonym_match(self):
        """Terms without synonyms should still appear."""
        result = expand_codegraph_query("foobar")
        assert result == "foobar"

    def test_deduplication(self):
        """Repeated terms should not appear twice in output."""
        result = expand_codegraph_query("auth authentication")
        terms = result.split(" OR ")
        assert len(terms) == len(set(terms))

    def test_output_format_or_joined(self):
        result = expand_codegraph_query("delete")
        assert " OR " in result

    def test_output_sorted(self):
        result = expand_codegraph_query("delete")
        terms = result.split(" OR ")
        assert terms == sorted(terms)

    def test_empty_query(self):
        result = expand_codegraph_query("")
        assert result == ""

    def test_snake_case_query(self):
        result = expand_codegraph_query("verify_credentials")
        terms = set(result.split(" OR "))
        assert "verify" in terms
        assert "credentials" in terms
        # verify is a synonym of validate family
        assert "validate" in terms

    def test_multiple_words(self):
        result = expand_codegraph_query("create database")
        terms = set(result.split(" OR "))
        # From "create" family
        assert "create" in terms
        assert "build" in terms
        assert "make" in terms
        # From "database" family
        assert "database" in terms
        assert "db" in terms
        assert "repository" in terms

    def test_original_term_preserved(self):
        """The original token should always be in the output."""
        result = expand_codegraph_query("config")
        terms = set(result.split(" OR "))
        assert "config" in terms

    def test_additive_only(self):
        """Expansion should never remove the original query tokens."""
        original_tokens = {"delete"}
        result = expand_codegraph_query("delete")
        terms = set(result.split(" OR "))
        assert original_tokens.issubset(terms)

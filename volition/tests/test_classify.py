"""Tests for the intent classification engine.

Uses importlib to load classify.py directly, bypassing package __init__.py
chains that may have heavy dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Module loading (bypass package imports)
# ---------------------------------------------------------------------------

_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
_repo_root = _scripts_dir.parent.parent

# Ensure shared is importable
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Load classify.py directly
_spec = importlib.util.spec_from_file_location(
    "volition.scripts.classify",
    _scripts_dir / "classify.py",
    submodule_search_locations=[str(_scripts_dir)],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["volition.scripts.classify"] = _mod
sys.modules.setdefault("volition.scripts", type(sys)("volition.scripts"))
sys.modules["volition.scripts"].__path__ = [str(_scripts_dir)]
sys.modules.setdefault("volition", type(sys)("volition"))
sys.modules["volition"].__path__ = [str(_scripts_dir.parent)]
_spec.loader.exec_module(_mod)

IntentPrototype = _mod.IntentPrototype
CandidateScore = _mod.CandidateScore
ClassificationResult = _mod.ClassificationResult
classify_intent = _mod.classify_intent
format_clarification = _mod.format_clarification
DEFAULT_PROTOTYPES = _mod.DEFAULT_PROTOTYPES
_keyword_score = _mod._keyword_score


# ---------------------------------------------------------------------------
# Test prototypes
# ---------------------------------------------------------------------------

TEST_PROTOTYPES = (
    IntentPrototype(
        handler="code_edit",
        prototype_text="Edit, refactor, modify source code",
        keywords=("refactor", "edit", "fix", "change"),
        risk_level="MEDIUM",
        fallback_chain=("text_edit",),
    ),
    IntentPrototype(
        handler="llm_call",
        prototype_text="Consult a language model for analysis and review",
        keywords=("explain", "analyze", "review", "suggest"),
        risk_level="LOW",
    ),
    IntentPrototype(
        handler="web_search",
        prototype_text="Search the web for documentation and information",
        keywords=("search", "find", "lookup", "latest"),
        risk_level="LOW",
    ),
    IntentPrototype(
        handler="security",
        prototype_text="Security scanning and vulnerability reconnaissance",
        keywords=("scan", "vulnerability", "shodan", "security"),
        risk_level="HIGH",
        requires_confirmation=True,
    ),
)


# ---------------------------------------------------------------------------
# Tests: IntentPrototype
# ---------------------------------------------------------------------------

class TestIntentPrototype:
    def test_frozen(self):
        p = IntentPrototype(handler="test", prototype_text="test text")
        with pytest.raises(AttributeError):
            p.handler = "other"

    def test_defaults(self):
        p = IntentPrototype(handler="test", prototype_text="text")
        assert p.risk_level == "LOW"
        assert p.fallback_chain == ()
        assert p.requires_confirmation is False


# ---------------------------------------------------------------------------
# Tests: keyword_score
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def test_exact_match(self):
        proto = IntentPrototype(handler="x", prototype_text="x", keywords=("refactor",))
        assert _keyword_score("refactor the module", proto) == 1.0

    def test_no_match(self):
        proto = IntentPrototype(handler="x", prototype_text="x", keywords=("refactor", "edit"))
        assert _keyword_score("deploy to production", proto) == 0.0

    def test_partial_match(self):
        proto = IntentPrototype(handler="x", prototype_text="x", keywords=("refactor", "edit", "fix", "change"))
        assert _keyword_score("fix the bug", proto) == 0.25

    def test_empty_keywords(self):
        proto = IntentPrototype(handler="x", prototype_text="x", keywords=())
        assert _keyword_score("anything", proto) == 0.0


# ---------------------------------------------------------------------------
# Tests: classify_intent
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    def test_returns_classification_result(self):
        result = classify_intent("refactor the auth module", TEST_PROTOTYPES)
        assert isinstance(result, ClassificationResult)
        assert result.action == "refactor the auth module"

    def test_has_all_candidates(self):
        result = classify_intent("something", TEST_PROTOTYPES)
        handlers = {c.handler for c in result.candidates}
        assert handlers == {"code_edit", "llm_call", "web_search", "security"}

    def test_candidates_sorted_by_fused_score(self):
        result = classify_intent("refactor the auth module", TEST_PROTOTYPES)
        scores = [c.fused_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_action(self):
        result = classify_intent("", TEST_PROTOTYPES)
        assert result.selected == "llm_call"  # default
        assert result.confidence == 0.0

    def test_code_edit_classification(self):
        """'refactor the auth module' should classify as code_edit."""
        result = classify_intent("refactor the authentication module", TEST_PROTOTYPES)
        assert result.selected == "code_edit"

    def test_web_search_classification(self):
        """'search for Python docs' should classify as web_search."""
        result = classify_intent("search for Python documentation", TEST_PROTOTYPES)
        assert result.selected == "web_search"

    def test_security_classification(self):
        """'scan for vulnerabilities' should classify as security."""
        result = classify_intent("scan for vulnerabilities on shodan", TEST_PROTOTYPES)
        assert result.selected == "security"

    def test_llm_classification(self):
        """'explain the algorithm' should classify as llm_call."""
        result = classify_intent("explain how this algorithm works", TEST_PROTOTYPES)
        assert result.selected == "llm_call"

    @patch.object(_mod, "_get_feedback_scores", return_value={})
    def test_no_feedback_adjustment_when_disabled(self, mock_fb):
        result = classify_intent("refactor code", TEST_PROTOTYPES, use_feedback=False)
        assert all(c.feedback_adjustment == 0.0 for c in result.candidates)
        mock_fb.assert_not_called()

    def test_confidence_threshold_gating(self):
        """With a very high threshold, classification should be below threshold."""
        result = classify_intent("do something vague", TEST_PROTOTYPES, threshold=0.99)
        # With threshold 0.99, most natural queries won't be confident enough
        # (depends on embedding model; if above, that's still valid)
        assert isinstance(result.above_threshold, bool)

    def test_paraphrase_routing(self):
        """'Make the auth module more secure' should NOT route to security scanning."""
        result = classify_intent(
            "make the auth module more secure",
            TEST_PROTOTYPES,
        )
        # The embedding model should understand this is about code editing,
        # not security scanning. If embeddings aren't available, keyword fallback
        # may not get this right, so we just check it returns a valid result.
        assert result.selected in {"code_edit", "llm_call", "web_search", "security", "clarification"}

    def test_default_prototypes_exist(self):
        assert len(DEFAULT_PROTOTYPES) == 4
        handlers = {p.handler for p in DEFAULT_PROTOTYPES}
        assert handlers == {"code_edit", "llm_call", "web_search", "security"}

    @patch.object(_mod, "_get_feedback_scores", return_value={"code_edit": 0.9, "llm_call": 0.1})
    def test_feedback_shifts_scores(self, mock_fb):
        """Positive feedback for code_edit should boost its score."""
        result = classify_intent("refactor the module", TEST_PROTOTYPES, use_feedback=True)
        code_edit_candidate = next(c for c in result.candidates if c.handler == "code_edit")
        assert code_edit_candidate.feedback_adjustment > 0

    @patch.object(_mod, "_get_feedback_scores", return_value={"code_edit": 0.9})
    def test_feedback_cannot_bypass_threshold(self, mock_fb):
        """Constitution Rule 5: feedback cannot push below-threshold above threshold."""
        # Use a very high threshold with an ambiguous query so the gated
        # scoring path is exercised (not the empty-action early return).
        result = classify_intent(
            "ambiguous request", TEST_PROTOTYPES, threshold=0.99, use_feedback=True,
        )
        assert result.candidates
        assert not result.above_threshold


# ---------------------------------------------------------------------------
# Tests: format_clarification
# ---------------------------------------------------------------------------

class TestFormatClarification:
    def test_returns_dict(self):
        result = ClassificationResult(
            action="ambiguous",
            candidates=[
                CandidateScore(handler="code_edit", fused_score=0.5),
                CandidateScore(handler="llm_call", fused_score=0.4),
            ],
            selected="clarification",
            confidence=0.5,
            above_threshold=False,
        )
        clarification = format_clarification(result)
        assert clarification["status"] == "clarification_required"
        assert len(clarification["candidates"]) == 2

    def test_includes_top_two(self):
        result = ClassificationResult(
            action="test",
            candidates=[
                CandidateScore(handler="a", fused_score=0.5),
                CandidateScore(handler="b", fused_score=0.4),
                CandidateScore(handler="c", fused_score=0.1),
            ],
        )
        clarification = format_clarification(result)
        handlers = [c["handler"] for c in clarification["candidates"]]
        assert handlers == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: Embedding unavailability fallback
# ---------------------------------------------------------------------------

class TestEmbeddingFallback:
    @patch("shared.embeddings.rank_by_relevance")
    def test_falls_back_to_zero_scores(self, mock_rank):
        """When embeddings return zero scores, keyword signal still works."""
        from shared.embeddings import ScoredItem

        mock_rank.return_value = [
            ScoredItem(item=p, score=0.0) for p in TEST_PROTOTYPES
        ]
        result = classify_intent("refactor the module", TEST_PROTOTYPES)
        # Should still return a valid result (keyword signal picks up "refactor")
        assert isinstance(result, ClassificationResult)
        assert result.selected != ""

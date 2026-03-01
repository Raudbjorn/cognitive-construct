"""Tests for outcome learning and feedback integration (Phase 5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
_repo_root = _scripts_dir.parent.parent

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Load classify
if "volition.scripts.classify" not in sys.modules:
    _cspec = importlib.util.spec_from_file_location(
        "volition.scripts.classify", _scripts_dir / "classify.py",
        submodule_search_locations=[str(_scripts_dir)],
    )
    _cmod = importlib.util.module_from_spec(_cspec)
    sys.modules["volition.scripts.classify"] = _cmod
    sys.modules.setdefault("volition.scripts", type(sys)("volition.scripts"))
    sys.modules["volition.scripts"].__path__ = [str(_scripts_dir)]
    sys.modules.setdefault("volition", type(sys)("volition"))
    sys.modules["volition"].__path__ = [str(_scripts_dir.parent)]
    _cspec.loader.exec_module(_cmod)
else:
    _cmod = sys.modules["volition.scripts.classify"]

classify_intent = _cmod.classify_intent
IntentPrototype = _cmod.IntentPrototype

TEST_PROTOTYPES = (
    IntentPrototype(
        handler="code_edit",
        prototype_text="Edit and refactor source code",
        keywords=("refactor", "edit", "fix"),
    ),
    IntentPrototype(
        handler="llm_call",
        prototype_text="Consult a language model",
        keywords=("explain", "analyze", "review"),
    ),
)


class TestFeedbackShiftsScores:
    @patch.object(_cmod, "_get_feedback_scores", return_value={"code_edit": 0.9, "llm_call": 0.1})
    def test_positive_feedback_boosts(self, mock_fb):
        result = classify_intent("refactor the module", TEST_PROTOTYPES, use_feedback=True)
        ce = next(c for c in result.candidates if c.handler == "code_edit")
        assert ce.feedback_adjustment > 0

    @patch.object(_cmod, "_get_feedback_scores", return_value={"code_edit": 0.1, "llm_call": 0.9})
    def test_negative_feedback_reduces(self, mock_fb):
        result = classify_intent("refactor the module", TEST_PROTOTYPES, use_feedback=True)
        ce = next(c for c in result.candidates if c.handler == "code_edit")
        assert ce.feedback_adjustment < 0

    @patch.object(_cmod, "_get_feedback_scores", return_value={"code_edit": 0.5})
    def test_neutral_feedback_no_change(self, mock_fb):
        result = classify_intent("refactor the module", TEST_PROTOTYPES, use_feedback=True)
        ce = next(c for c in result.candidates if c.handler == "code_edit")
        assert abs(ce.feedback_adjustment) < 0.01


class TestSafetyConstraintsNotBypassed:
    @patch.object(_cmod, "_get_feedback_scores", return_value={"code_edit": 1.0})
    def test_threshold_not_bypassed_by_feedback(self, mock_fb):
        """Constitution Rule 5: raw fused score must pass threshold independently."""
        # With only 2 prototypes, RRF normalization pushes top score to 1.0,
        # so we use threshold > 1.0 to guarantee below-threshold before feedback.
        result = classify_intent(
            "perform a vague task", TEST_PROTOTYPES, threshold=1.01, use_feedback=True,
        )
        assert result.candidates
        assert not result.above_threshold

    @patch.object(_cmod, "_get_feedback_scores", return_value={})
    def test_no_feedback_same_as_neutral(self, mock_fb):
        result_no_fb = classify_intent("refactor code", TEST_PROTOTYPES, use_feedback=False)
        result_with_fb = classify_intent("refactor code", TEST_PROTOTYPES, use_feedback=True)
        # Both should select the same handler when feedback returns empty
        assert result_no_fb.selected == result_with_fb.selected


class TestLearningRateEnv:
    @patch.object(_cmod, "_LEARNING_RATE", 0.0)
    @patch.object(_cmod, "_get_feedback_scores", return_value={"code_edit": 1.0})
    def test_zero_learning_rate_no_adjustment(self, mock_fb):
        result = classify_intent("refactor code", TEST_PROTOTYPES, use_feedback=True)
        ce = next(c for c in result.candidates if c.handler == "code_edit")
        assert abs(ce.feedback_adjustment) < 0.001

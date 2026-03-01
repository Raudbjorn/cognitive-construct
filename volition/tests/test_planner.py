"""Tests for action plan construction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
_repo_root = _scripts_dir.parent.parent

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Ensure classify is loaded first (planner imports it)
if "volition.scripts.classify" not in sys.modules:
    _cspec = importlib.util.spec_from_file_location(
        "volition.scripts.classify",
        _scripts_dir / "classify.py",
        submodule_search_locations=[str(_scripts_dir)],
    )
    _cmod = importlib.util.module_from_spec(_cspec)
    sys.modules["volition.scripts.classify"] = _cmod
    sys.modules.setdefault("volition.scripts", type(sys)("volition.scripts"))
    sys.modules["volition.scripts"].__path__ = [str(_scripts_dir)]
    sys.modules.setdefault("volition", type(sys)("volition"))
    sys.modules["volition"].__path__ = [str(_scripts_dir.parent)]
    _cspec.loader.exec_module(_cmod)

# Load planner.py
_pspec = importlib.util.spec_from_file_location(
    "volition.scripts.planner",
    _scripts_dir / "planner.py",
    submodule_search_locations=[str(_scripts_dir)],
)
_pmod = importlib.util.module_from_spec(_pspec)
sys.modules["volition.scripts.planner"] = _pmod
# Need to set .classify on the module for relative import
_pmod.classify = sys.modules["volition.scripts.classify"]
_pspec.loader.exec_module(_pmod)

PlanStep = _pmod.PlanStep
ActionPlan = _pmod.ActionPlan
ActionOutcome = _pmod.ActionOutcome
split_compound_action = _pmod.split_compound_action
topological_sort = _pmod.topological_sort
CyclicDependencyError = _pmod.CyclicDependencyError
resolve_inputs = _pmod.resolve_inputs
build_single_step_plan = _pmod.build_single_step_plan
build_multi_step_plan = _pmod.build_multi_step_plan
build_plan = _pmod.build_plan

ClassificationResult = sys.modules["volition.scripts.classify"].ClassificationResult
CandidateScore = sys.modules["volition.scripts.classify"].CandidateScore


# ---------------------------------------------------------------------------
# Tests: split_compound_action
# ---------------------------------------------------------------------------

class TestSplitCompoundAction:
    def test_simple_action(self):
        parts = split_compound_action("refactor the auth module")
        assert parts == ["refactor the auth module"]

    def test_and_conjunction(self):
        parts = split_compound_action("review the code and fix any issues")
        assert len(parts) == 2
        assert "review the code" in parts[0]
        assert "fix any issues" in parts[1]

    def test_then_conjunction(self):
        parts = split_compound_action("analyze the module then suggest improvements")
        assert len(parts) == 2

    def test_and_then_conjunction(self):
        parts = split_compound_action("review the code and then apply fixes")
        assert len(parts) == 2

    def test_multiple_conjunctions(self):
        parts = split_compound_action("search for docs and review them then summarize")
        assert len(parts) == 3

    def test_empty_string(self):
        parts = split_compound_action("")
        assert parts == [""]


# ---------------------------------------------------------------------------
# Tests: topological_sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_single_step(self):
        steps = [PlanStep(step_id="s1", handler="llm_call", action="test")]
        result = topological_sort(steps)
        assert len(result) == 1
        assert result[0].step_id == "s1"

    def test_linear_chain(self):
        steps = [
            PlanStep(step_id="s1", handler="llm_call", action="review"),
            PlanStep(step_id="s2", handler="code_edit", action="fix", depends_on=["s1"]),
        ]
        result = topological_sort(steps)
        assert result[0].step_id == "s1"
        assert result[1].step_id == "s2"

    def test_circular_dependency_raises(self):
        steps = [
            PlanStep(step_id="s1", handler="a", action="x", depends_on=["s2"]),
            PlanStep(step_id="s2", handler="b", action="y", depends_on=["s1"]),
        ]
        with pytest.raises(CyclicDependencyError):
            topological_sort(steps)

    def test_independent_steps(self):
        steps = [
            PlanStep(step_id="s1", handler="a", action="x"),
            PlanStep(step_id="s2", handler="b", action="y"),
        ]
        result = topological_sort(steps)
        assert len(result) == 2

    def test_diamond_dependency(self):
        steps = [
            PlanStep(step_id="s1", handler="a", action="root"),
            PlanStep(step_id="s2", handler="b", action="left", depends_on=["s1"]),
            PlanStep(step_id="s3", handler="c", action="right", depends_on=["s1"]),
            PlanStep(step_id="s4", handler="d", action="merge", depends_on=["s2", "s3"]),
        ]
        result = topological_sort(steps)
        ids = [r.step_id for r in result]
        assert ids.index("s1") < ids.index("s2")
        assert ids.index("s1") < ids.index("s3")
        assert ids.index("s2") < ids.index("s4")
        assert ids.index("s3") < ids.index("s4")


# ---------------------------------------------------------------------------
# Tests: resolve_inputs
# ---------------------------------------------------------------------------

class TestResolveInputs:
    def test_no_references(self):
        result = resolve_inputs({"query": "test"}, {})
        assert result == {"query": "test"}

    def test_step_reference(self):
        outputs = {"step-1": {"data": "review findings"}}
        result = resolve_inputs({"findings": "step-1.output"}, outputs)
        assert result["findings"] == {"data": "review findings"}

    def test_missing_reference_passthrough(self):
        result = resolve_inputs({"x": "step-999.output"}, {})
        assert result["x"] == "step-999.output"

    def test_non_reference_passthrough(self):
        result = resolve_inputs({"x": "plain value"}, {"step-1": "data"})
        assert result["x"] == "plain value"


# ---------------------------------------------------------------------------
# Tests: build_single_step_plan
# ---------------------------------------------------------------------------

class TestBuildSingleStepPlan:
    def test_creates_plan(self):
        classification = ClassificationResult(
            action="refactor code",
            selected="code_edit",
            confidence=0.8,
            above_threshold=True,
        )
        plan = build_single_step_plan("refactor code", classification)
        assert isinstance(plan, ActionPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].handler == "code_edit"
        assert plan.original_action == "refactor code"

    def test_includes_fallback_chain(self):
        classification = ClassificationResult(
            action="edit code",
            selected="code_edit",
            confidence=0.8,
            above_threshold=True,
        )
        plan = build_single_step_plan("edit code", classification)
        assert plan.steps[0].fallback_chain == ["text_edit"]


# ---------------------------------------------------------------------------
# Tests: build_plan (auto-detection)
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_single_step_simple_action(self):
        plan = build_plan("refactor the auth module")
        assert len(plan.steps) == 1

    def test_multi_step_compound_action(self):
        plan = build_plan("review the auth module and fix any issues")
        assert len(plan.steps) >= 2

    def test_multi_step_dependencies(self):
        plan = build_plan("analyze the code then apply improvements")
        if len(plan.steps) > 1:
            assert plan.steps[1].depends_on == ["step-1"]

    def test_plan_has_id(self):
        plan = build_plan("do something")
        assert plan.plan_id.startswith("plan-")

    def test_plan_has_timestamp(self):
        plan = build_plan("do something")
        assert plan.created is not None

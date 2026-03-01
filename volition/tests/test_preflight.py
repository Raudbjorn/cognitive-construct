"""Tests for the pre-flight validation engine."""

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

# Load planner
if "volition.scripts.planner" not in sys.modules:
    _pspec = importlib.util.spec_from_file_location(
        "volition.scripts.planner", _scripts_dir / "planner.py",
        submodule_search_locations=[str(_scripts_dir)],
    )
    _pmod = importlib.util.module_from_spec(_pspec)
    sys.modules["volition.scripts.planner"] = _pmod
    _pspec.loader.exec_module(_pmod)

# Load preflight
_pfspec = importlib.util.spec_from_file_location(
    "volition.scripts.preflight", _scripts_dir / "preflight.py",
    submodule_search_locations=[str(_scripts_dir)],
)
_pfmod = importlib.util.module_from_spec(_pfspec)
sys.modules["volition.scripts.preflight"] = _pfmod
_pfspec.loader.exec_module(_pfmod)

PreflightFlag = _pfmod.PreflightFlag
PassResult = _pfmod.PassResult
PreflightResult = _pfmod.PreflightResult
pass_capability = _pfmod.pass_capability
pass_input_validation = _pfmod.pass_input_validation
pass_risk_assessment = _pfmod.pass_risk_assessment
pass_dependency_check = _pfmod.pass_dependency_check
preflight_validate = _pfmod.preflight_validate

PlanStep = sys.modules["volition.scripts.planner"].PlanStep
ActionPlan = sys.modules["volition.scripts.planner"].ActionPlan


def _make_plan(*steps: PlanStep) -> ActionPlan:
    return ActionPlan(plan_id="test-plan", steps=list(steps))


# ---------------------------------------------------------------------------
# Tests: Pass 1 - Capability check
# ---------------------------------------------------------------------------

class TestPassCapability:
    def test_all_available(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))
        caps = {"llm_consultation": {"status": "available"}}
        result = pass_capability(plan, caps)
        assert result.status == "pass"

    def test_unavailable_no_fallback(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="code_edit", action="test"))
        caps = {"code_editing": {"status": "unavailable"}}
        result = pass_capability(plan, caps)
        assert result.status == "flagged"
        assert any(f.severity == "CRITICAL" for f in result.flags)

    def test_unavailable_with_fallback(self):
        plan = _make_plan(PlanStep(
            step_id="s1", handler="code_edit", action="test",
            fallback_chain=["text_edit"],
        ))
        caps = {"code_editing": {"status": "unavailable"}}
        result = pass_capability(plan, caps)
        assert result.status == "flagged"
        assert all(f.severity == "WARNING" for f in result.flags)

    def test_unknown_handler(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="unknown", action="test"))
        result = pass_capability(plan, {})
        assert any(f.type == "unknown_handler" for f in result.flags)


# ---------------------------------------------------------------------------
# Tests: Pass 2 - Input validation
# ---------------------------------------------------------------------------

class TestPassInputValidation:
    def test_valid_plan(self):
        plan = _make_plan(
            PlanStep(step_id="s1", handler="llm_call", action="review"),
            PlanStep(step_id="s2", handler="code_edit", action="fix",
                     inputs={"findings": "s1.output"}),
        )
        result = pass_input_validation(plan)
        assert result.status == "pass"

    def test_empty_action(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action=""))
        result = pass_input_validation(plan)
        assert any(f.type == "empty_action" for f in result.flags)

    def test_invalid_step_reference(self):
        plan = _make_plan(PlanStep(
            step_id="s1", handler="llm_call", action="test",
            inputs={"x": "nonexistent.output"},
        ))
        result = pass_input_validation(plan)
        assert any(f.type == "invalid_step_reference" for f in result.flags)


# ---------------------------------------------------------------------------
# Tests: Pass 3 - Risk assessment
# ---------------------------------------------------------------------------

class TestPassRiskAssessment:
    def test_security_without_confirm(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="security", action="scan"))
        result = pass_risk_assessment(plan, confirm=False)
        assert any(f.type == "unconfirmed_security_action" for f in result.flags)
        assert any(f.severity == "CRITICAL" for f in result.flags)

    def test_security_with_confirm(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="security", action="scan"))
        result = pass_risk_assessment(plan, confirm=True)
        assert result.status == "pass"

    def test_non_security_no_flags(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="analyze"))
        result = pass_risk_assessment(plan)
        assert result.status == "pass"

    def test_rhetoric_preflight_warning(self):
        plan = _make_plan(PlanStep(
            step_id="s1", handler="security", action="scan", risk_level="HIGH",
        ))
        result = pass_risk_assessment(plan, confirm=True, rhetoric_preflight=True)
        assert any(f.type == "rhetoric_preflight_required" for f in result.flags)


# ---------------------------------------------------------------------------
# Tests: Pass 4 - Dependency check
# ---------------------------------------------------------------------------

class TestPassDependencyCheck:
    def test_valid_dag(self):
        plan = _make_plan(
            PlanStep(step_id="s1", handler="a", action="x"),
            PlanStep(step_id="s2", handler="b", action="y", depends_on=["s1"]),
        )
        result = pass_dependency_check(plan)
        assert result.status == "pass"

    def test_missing_dependency(self):
        plan = _make_plan(
            PlanStep(step_id="s1", handler="a", action="x", depends_on=["s99"]),
        )
        result = pass_dependency_check(plan)
        assert any(f.type == "missing_dependency" for f in result.flags)

    def test_circular_dependency(self):
        plan = _make_plan(
            PlanStep(step_id="s1", handler="a", action="x", depends_on=["s2"]),
            PlanStep(step_id="s2", handler="b", action="y", depends_on=["s1"]),
        )
        result = pass_dependency_check(plan)
        assert any(f.type == "circular_dependency" for f in result.flags)

    def test_forward_reference_input(self):
        plan = _make_plan(
            PlanStep(step_id="s1", handler="a", action="x",
                     inputs={"data": "s2.output"}),
            PlanStep(step_id="s2", handler="b", action="y"),
        )
        result = pass_dependency_check(plan)
        assert any(f.type == "forward_reference" for f in result.flags)


# ---------------------------------------------------------------------------
# Tests: Full pipeline
# ---------------------------------------------------------------------------

class TestPreflightValidate:
    def test_valid_plan(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))
        caps = {"llm_consultation": {"status": "available"}}
        result = preflight_validate(plan, capabilities=caps)
        assert result.status == "valid"
        assert not result.has_critical

    def test_blocked_on_critical(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="security", action="scan"))
        caps = {"security_queries": {"status": "available"}}
        result = preflight_validate(plan, capabilities=caps, confirm=False)
        assert result.status == "blocked"
        assert result.has_critical

    def test_passes_dict_populated(self):
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))
        result = preflight_validate(plan, capabilities={})
        assert "capability_check" in result.passes
        assert "input_validation" in result.passes
        assert "risk_assessment" in result.passes
        assert "dependency_check" in result.passes

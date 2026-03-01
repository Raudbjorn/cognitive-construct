"""Tests for the execution engine."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

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

# Load handlers (no submodule_search_locations — leaf module)
if "volition.scripts.handlers" not in sys.modules:
    _hspec = importlib.util.spec_from_file_location(
        "volition.scripts.handlers", _scripts_dir / "handlers.py",
    )
    _hmod = importlib.util.module_from_spec(_hspec)
    _hmod.__package__ = "volition.scripts"
    sys.modules["volition.scripts.handlers"] = _hmod
    _hspec.loader.exec_module(_hmod)
else:
    _hmod = sys.modules["volition.scripts.handlers"]

# Load executor (no submodule_search_locations — leaf module, not a package)
if "volition.scripts.executor" not in sys.modules:
    _espec = importlib.util.spec_from_file_location(
        "volition.scripts.executor", _scripts_dir / "executor.py",
    )
    _emod = importlib.util.module_from_spec(_espec)
    _emod.__package__ = "volition.scripts"
    sys.modules["volition.scripts.executor"] = _emod
    _espec.loader.exec_module(_emod)
else:
    _emod = sys.modules["volition.scripts.executor"]

execute_plan = _emod.execute_plan
PlanStep = sys.modules["volition.scripts.planner"].PlanStep
ActionPlan = sys.modules["volition.scripts.planner"].ActionPlan
ActionOutcome = sys.modules["volition.scripts.planner"].ActionOutcome
HandlerConfig = sys.modules["volition.scripts.handlers"].HandlerConfig
register_handler = sys.modules["volition.scripts.handlers"].register_handler
clear_registry = sys.modules["volition.scripts.handlers"].clear_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(*steps: PlanStep) -> ActionPlan:
    return ActionPlan(plan_id="test-plan", original_action="test action", steps=list(steps))


def _setup_handler(name: str, return_value: dict | None = None, raises: bool = False):
    """Register a handler with a mock dispatch function."""
    if raises:
        async def dispatch(action, inputs):
            raise RuntimeError("boom")
    else:
        rv = return_value or {"status": "success", "summary": f"{name} completed"}
        async def dispatch(action, inputs):
            return rv
    register_handler(HandlerConfig(name=name, dispatch_fn=dispatch))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecutePlan:
    def setup_method(self):
        clear_registry()

    @pytest.mark.asyncio
    async def test_single_step_success(self, tmp_path):
        _setup_handler("llm_call")
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert len(outcomes) == 1
        assert outcomes[0].status == "success"
        assert outcomes[0].step_id == "s1"

    @pytest.mark.asyncio
    async def test_multi_step_success(self, tmp_path):
        _setup_handler("llm_call")
        _setup_handler("code_edit")
        plan = _make_plan(
            PlanStep(step_id="s1", handler="llm_call", action="review"),
            PlanStep(step_id="s2", handler="code_edit", action="fix", depends_on=["s1"]),
        )

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert len(outcomes) == 2
        assert all(o.status == "success" for o in outcomes)

    @pytest.mark.asyncio
    async def test_abort_on_failure(self, tmp_path):
        """Constitution Rule 6: abort on step failure."""
        _setup_handler("llm_call", return_value={"status": "error", "message": "failed"})
        _setup_handler("code_edit")
        plan = _make_plan(
            PlanStep(step_id="s1", handler="llm_call", action="review"),
            PlanStep(step_id="s2", handler="code_edit", action="fix", depends_on=["s1"]),
        )

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert len(outcomes) == 1  # s2 never executed
        assert outcomes[0].status == "error"

    @pytest.mark.asyncio
    async def test_fallback_chain(self, tmp_path):
        _setup_handler("code_edit", return_value={"status": "error", "message": "serena down"})
        _setup_handler("text_edit", return_value={"status": "success", "summary": "text edit ok"})

        plan = _make_plan(PlanStep(
            step_id="s1", handler="code_edit", action="fix code",
            fallback_chain=["text_edit"],
        ))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert outcomes[0].status == "success"
        assert outcomes[0].fallbacks_attempted == ["text_edit"]

    @pytest.mark.asyncio
    async def test_unregistered_handler(self, tmp_path):
        plan = _make_plan(PlanStep(step_id="s1", handler="nonexistent", action="test"))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert outcomes[0].status == "error"

    @pytest.mark.asyncio
    async def test_handler_exception(self, tmp_path):
        _setup_handler("llm_call", raises=True)
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert outcomes[0].status == "error"

    @pytest.mark.asyncio
    async def test_audit_log_written(self, tmp_path):
        """Constitution Rule 7: log before acting."""
        _setup_handler("llm_call")
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))
        audit_file = tmp_path / "audit.log"

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", audit_file):
            await execute_plan(plan)

        assert audit_file.exists()
        lines = audit_file.read_text().strip().split("\n")
        # Should have at least: started entry, outcome entry, completed entry
        assert len(lines) >= 3

    @pytest.mark.asyncio
    async def test_feedback_recorded(self, tmp_path):
        _setup_handler("llm_call")
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert outcomes[0].feedback_recorded is True

    @pytest.mark.asyncio
    async def test_duration_tracked(self, tmp_path):
        _setup_handler("llm_call")
        plan = _make_plan(PlanStep(step_id="s1", handler="llm_call", action="test"))

        with patch.object(_emod, "VOLITION_DIR", tmp_path), \
             patch.object(_emod, "AUDIT_LOG", tmp_path / "audit.log"):
            outcomes = await execute_plan(plan)

        assert outcomes[0].duration_ms >= 0

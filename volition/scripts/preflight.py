"""Pre-flight Validation Engine for Volition.

Four-pass validation mirroring Rhetoric's validate.py, adapted for action
plans instead of argument graphs. Pure functions, no I/O, deterministic.

See SPEC.md Section 3.4 for pass definitions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .planner import ActionPlan, CyclicDependencyError, PlanStep, topological_sort

# Ensure shared is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreflightFlag:
    """A validation issue found during pre-flight."""

    type: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    step_id: str
    description: str
    remediation: str = ""


@dataclass
class PassResult:
    """Result from a single validation pass."""

    status: str = "pass"  # "pass" or "flagged"
    flags: list[PreflightFlag] = field(default_factory=list)


@dataclass
class PreflightResult:
    """Full pre-flight validation result."""

    plan_id: str = ""
    status: str = "valid"  # "valid", "flagged", or "blocked"
    passes: dict[str, PassResult] = field(default_factory=dict)

    @property
    def all_flags(self) -> list[PreflightFlag]:
        """All flags across all passes."""
        return [f for p in self.passes.values() for f in p.flags]

    @property
    def has_critical(self) -> bool:
        """Whether any CRITICAL flag exists."""
        return any(f.severity == "CRITICAL" for f in self.all_flags)


# ---------------------------------------------------------------------------
# Handler → capability mapping
# ---------------------------------------------------------------------------

HANDLER_TO_CAPABILITY: dict[str, str] = {
    "code_edit": "code_editing",
    "text_edit": "code_editing",
    "llm_call": "llm_consultation",
    "web_search": "web_search",
    "security": "security_queries",
}

# Handlers that always require confirmation (Constitution Rule 2)
_CONFIRMATION_REQUIRED_HANDLERS = frozenset({"security"})

# Required input fields per handler type
_REQUIRED_INPUTS: dict[str, set[str]] = {
    "code_edit": set(),  # symbol and change extracted at dispatch time
    "llm_call": set(),
    "web_search": set(),
    "security": set(),
}


# ---------------------------------------------------------------------------
# Pass 1: Capability check
# ---------------------------------------------------------------------------

def _get_capabilities() -> dict[str, dict[str, Any]]:
    """Load capabilities, returning empty dict on failure."""
    try:
        from .volition import cmd_capabilities

        return cmd_capabilities()
    except Exception:
        return {}


def pass_capability(plan: ActionPlan, capabilities: dict[str, Any] | None = None) -> PassResult:
    """Pass 1: Verify each step's handler is available."""
    if capabilities is None:
        capabilities = _get_capabilities()

    flags: list[PreflightFlag] = []
    for step in plan.steps:
        cap_name = HANDLER_TO_CAPABILITY.get(step.handler)
        if cap_name is None:
            flags.append(PreflightFlag(
                type="unknown_handler",
                severity="CRITICAL",
                step_id=step.step_id,
                description=f"Handler '{step.handler}' is not recognized",
                remediation="Use a valid handler: code_edit, llm_call, web_search, security",
            ))
            continue

        cap = capabilities.get(cap_name, {})
        status = cap.get("status", "unavailable") if isinstance(cap, dict) else "unavailable"
        if status == "unavailable":
            has_fallback = bool(step.fallback_chain)
            flags.append(PreflightFlag(
                type="handler_unavailable",
                severity="WARNING" if has_fallback else "CRITICAL",
                step_id=step.step_id,
                description=f"Handler '{step.handler}' ({cap_name}) is not available",
                remediation=(
                    f"Fallback available: {step.fallback_chain}"
                    if has_fallback
                    else "No fallback configured"
                ),
            ))

    return PassResult(
        status="flagged" if flags else "pass",
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Pass 2: Input validation
# ---------------------------------------------------------------------------

def pass_input_validation(plan: ActionPlan) -> PassResult:
    """Pass 2: Verify step inputs are well-formed."""
    flags: list[PreflightFlag] = []
    step_ids = {s.step_id for s in plan.steps}

    for step in plan.steps:
        # Check action is non-empty
        if not step.action.strip():
            flags.append(PreflightFlag(
                type="empty_action",
                severity="CRITICAL",
                step_id=step.step_id,
                description="Step has empty action text",
                remediation="Provide a non-empty action description",
            ))

        # Check step references in inputs resolve
        for key, value in step.inputs.items():
            if isinstance(value, str) and "." in value:
                ref_step = value.split(".")[0]
                if ref_step not in step_ids:
                    flags.append(PreflightFlag(
                        type="invalid_step_reference",
                        severity="CRITICAL",
                        step_id=step.step_id,
                        description=f"Input '{key}' references non-existent step '{ref_step}'",
                        remediation=f"Valid steps: {sorted(step_ids)}",
                    ))

        # Check required inputs for handler type
        required = _REQUIRED_INPUTS.get(step.handler, set())
        missing = required - set(step.inputs.keys())
        if missing:
            flags.append(PreflightFlag(
                type="missing_required_input",
                severity="CRITICAL",
                step_id=step.step_id,
                description=f"Missing required inputs for {step.handler}: {missing}",
                remediation=f"Add required inputs: {sorted(missing)}",
            ))

    return PassResult(
        status="flagged" if flags else "pass",
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Pass 3: Risk assessment
# ---------------------------------------------------------------------------

def pass_risk_assessment(
    plan: ActionPlan,
    confirm: bool = False,
    rhetoric_preflight: bool = False,
) -> PassResult:
    """Pass 3: Check safety constraints.

    Args:
        plan: The action plan to validate.
        confirm: Whether --confirm was provided.
        rhetoric_preflight: Whether RHETORIC_PREFLIGHT is enabled.
    """
    flags: list[PreflightFlag] = []

    for step in plan.steps:
        # Constitution Rule 2: security actions always require confirmation
        if step.handler in _CONFIRMATION_REQUIRED_HANDLERS and not confirm:
            flags.append(PreflightFlag(
                type="unconfirmed_security_action",
                severity="CRITICAL",
                step_id=step.step_id,
                description=(
                    f"Handler '{step.handler}' requires explicit confirmation"
                ),
                remediation="Add --confirm flag",
            ))

        # High-risk steps trigger rhetoric preflight if enabled
        if step.risk_level == "HIGH" and rhetoric_preflight:
            flags.append(PreflightFlag(
                type="rhetoric_preflight_required",
                severity="WARNING",
                step_id=step.step_id,
                description="High-risk step requires Rhetoric deliberation",
                remediation="RHETORIC_PREFLIGHT is enabled; deliberation will be requested",
            ))

    return PassResult(
        status="flagged" if flags else "pass",
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Pass 4: Dependency check
# ---------------------------------------------------------------------------

def pass_dependency_check(plan: ActionPlan) -> PassResult:
    """Pass 4: Validate the DAG structure."""
    flags: list[PreflightFlag] = []
    step_ids = {s.step_id for s in plan.steps}

    # Check all depends_on references point to existing steps
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                flags.append(PreflightFlag(
                    type="missing_dependency",
                    severity="CRITICAL",
                    step_id=step.step_id,
                    description=f"Depends on non-existent step '{dep}'",
                    remediation=f"Valid steps: {sorted(step_ids)}",
                ))

    # Check for circular dependencies by reusing planner.topological_sort
    try:
        ordered_steps = topological_sort(plan.steps)
        ordered = [s.step_id for s in ordered_steps]
    except CyclicDependencyError as e:
        flags.append(PreflightFlag(
            type="circular_dependency",
            severity="CRITICAL",
            step_id=str(e).split(":")[-1].strip().split(",")[0].strip(" '[]") if str(e) else "unknown",
            description=str(e),
            remediation="Remove dependency cycle",
        ))
        ordered = [s.step_id for s in plan.steps]  # best-effort ordering

    # Check input references are from prior steps (topological ordering)
    position = {sid: idx for idx, sid in enumerate(ordered)}
    for step in plan.steps:
        step_pos = position.get(step.step_id, -1)
        for key, value in step.inputs.items():
            if isinstance(value, str) and "." in value:
                ref_step = value.split(".")[0]
                ref_pos = position.get(ref_step, -1)
                if ref_pos >= step_pos and ref_step in step_ids:
                    flags.append(PreflightFlag(
                        type="forward_reference",
                        severity="CRITICAL",
                        step_id=step.step_id,
                        description=(
                            f"Input '{key}' references step '{ref_step}' "
                            f"which is not earlier in execution order"
                        ),
                        remediation="Inputs can only reference steps that execute before this one",
                    ))

    return PassResult(
        status="flagged" if flags else "pass",
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def preflight_validate(
    plan: ActionPlan,
    capabilities: dict[str, Any] | None = None,
    confirm: bool = False,
) -> PreflightResult:
    """Run all four pre-flight validation passes.

    Constitution Rule 3: CRITICAL flags block execution.
    """
    # Check RHETORIC_PREFLIGHT feature flag
    rhetoric_preflight = False
    try:
        from shared.feature_flags import get_flag

        rhetoric_preflight = get_flag("RHETORIC_PREFLIGHT", False)
    except Exception:
        pass

    passes = {
        "capability_check": pass_capability(plan, capabilities),
        "input_validation": pass_input_validation(plan),
        "risk_assessment": pass_risk_assessment(plan, confirm, rhetoric_preflight),
        "dependency_check": pass_dependency_check(plan),
    }

    result = PreflightResult(
        plan_id=plan.plan_id,
        passes=passes,
    )

    if result.has_critical:
        result.status = "blocked"
    elif result.all_flags:
        result.status = "flagged"
    else:
        result.status = "valid"

    return result

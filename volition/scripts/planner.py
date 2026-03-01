"""Action Plan Construction for Volition.

Builds single-step and multi-step DAGs from classified intents.
Steps have explicit handlers, input bindings, and fallback chains.

See SPEC.md Section 3.2.3 for ActionPlan structure and Section 5 Phase 2.
"""

from __future__ import annotations

import re
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .classify import ClassificationResult, IntentPrototype, classify_intent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an action plan."""

    step_id: str
    handler: str
    action: str
    inputs: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    risk_level: str = "LOW"
    fallback_chain: list[str] = field(default_factory=list)


@dataclass
class ActionPlan:
    """Directed acyclic graph of execution steps."""

    plan_id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    original_action: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    classification: ClassificationResult | None = None


@dataclass
class ActionOutcome:
    """Result of executing a single step."""

    plan_id: str
    step_id: str
    handler: str
    status: str  # "success", "error", "skipped"
    duration_ms: int = 0
    output_summary: str = ""
    fallbacks_attempted: list[str] = field(default_factory=list)
    feedback_recorded: bool = False
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conjunction splitting
# ---------------------------------------------------------------------------

_CONJUNCTION_PATTERN = re.compile(
    r"\b(?:and then|then|after that|followed by|and)\b",
    re.IGNORECASE,
)


def split_compound_action(action: str) -> list[str]:
    """Split a compound action into sub-actions using conjunction detection.

    Simple deterministic splitting per SPEC Q3: no LLM decomposition.
    """
    parts = _CONJUNCTION_PATTERN.split(action)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned if cleaned else [action]


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

class CyclicDependencyError(ValueError):
    """Raised when an ActionPlan contains circular dependencies."""


def topological_sort(steps: list[PlanStep]) -> list[PlanStep]:
    """Sort steps in dependency order (Kahn's algorithm).

    Raises:
        CyclicDependencyError: If the DAG contains a cycle.
    """
    step_map = {s.step_id: s for s in steps}
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in steps}

    for step in steps:
        for dep in step.depends_on:
            if dep not in step_map:
                continue
            adjacency[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue: deque[str] = deque(
        sid for sid, deg in in_degree.items() if deg == 0
    )
    result: list[PlanStep] = []

    while queue:
        sid = queue.popleft()
        result.append(step_map[sid])
        for neighbor in adjacency[sid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(steps):
        raise CyclicDependencyError(
            f"Circular dependency detected among steps: "
            f"{[s.step_id for s in steps if s.step_id not in {r.step_id for r in result}]}"
        )

    return result


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_inputs(
    inputs: dict[str, str],
    step_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Replace step references (e.g. "step-1.output") with actual values."""
    resolved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, str) and "." in value:
            parts = value.split(".", 1)
            ref_step = parts[0]
            if ref_step in step_outputs:
                resolved[key] = step_outputs[ref_step]
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------

def _prototype_for_handler(
    handler: str,
    prototypes: tuple[IntentPrototype, ...],
) -> IntentPrototype | None:
    """Find prototype by handler name."""
    for p in prototypes:
        if p.handler == handler:
            return p
    return None


def build_single_step_plan(
    action: str,
    classification: ClassificationResult,
    prototypes: tuple[IntentPrototype, ...] | None = None,
) -> ActionPlan:
    """Build a single-step plan wrapping the classified handler."""
    from .classify import DEFAULT_PROTOTYPES

    if prototypes is None:
        prototypes = DEFAULT_PROTOTYPES

    proto = _prototype_for_handler(classification.selected, prototypes)
    risk = proto.risk_level if proto else "LOW"
    fallbacks = list(proto.fallback_chain) if proto else []

    step = PlanStep(
        step_id="step-1",
        handler=classification.selected,
        action=action,
        risk_level=risk,
        fallback_chain=fallbacks,
    )

    return ActionPlan(
        original_action=action,
        steps=[step],
        classification=classification,
    )


def build_multi_step_plan(
    action: str,
    prototypes: tuple[IntentPrototype, ...] | None = None,
    threshold: float | None = None,
) -> ActionPlan:
    """Build a multi-step plan by splitting compound actions.

    Each sub-action is independently classified and chained with
    data dependencies. Callers should use build_plan() which auto-detects
    single vs multi-step.
    """
    from .classify import DEFAULT_PROTOTYPES

    if prototypes is None:
        prototypes = DEFAULT_PROTOTYPES

    sub_actions = split_compound_action(action)
    steps: list[PlanStep] = []
    full_classification: ClassificationResult | None = None

    for i, sub_action in enumerate(sub_actions, start=1):
        classification = classify_intent(sub_action, prototypes, threshold)
        if i == 1:
            full_classification = classification

        proto = _prototype_for_handler(classification.selected, prototypes)
        risk = proto.risk_level if proto else "LOW"
        fallbacks = list(proto.fallback_chain) if proto else []

        step = PlanStep(
            step_id=f"step-{i}",
            handler=classification.selected,
            action=sub_action,
            depends_on=[f"step-{i - 1}"] if i > 1 else [],
            inputs=(
                {f"prior_output": f"step-{i - 1}.output"} if i > 1 else {}
            ),
            risk_level=risk,
            fallback_chain=fallbacks,
        )
        steps.append(step)

    return ActionPlan(
        original_action=action,
        steps=steps,
        classification=full_classification,
    )


def build_plan(
    action: str,
    prototypes: tuple[IntentPrototype, ...] | None = None,
    threshold: float | None = None,
) -> ActionPlan:
    """Top-level plan builder: auto-detects single vs multi-step.

    If the action contains conjunctions ("and then", "then", etc.),
    builds a multi-step plan. Otherwise, builds a single-step plan.
    """
    sub_actions = split_compound_action(action)
    if len(sub_actions) > 1:
        return build_multi_step_plan(action, prototypes, threshold)

    from .classify import DEFAULT_PROTOTYPES

    if prototypes is None:
        prototypes = DEFAULT_PROTOTYPES

    classification = classify_intent(action, prototypes, threshold)
    return build_single_step_plan(action, classification, prototypes)

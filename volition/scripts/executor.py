"""Execution Engine for Volition.

Processes validated ActionPlans step-by-step in topological order,
with fallback chain recovery and outcome logging.

See SPEC.md Sections 3.5, 3.6 and Phase 4-5.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .handlers import get_handler
from .planner import ActionOutcome, ActionPlan, PlanStep, resolve_inputs, topological_sort

logger = logging.getLogger(__name__)

# Ensure shared is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

VOLITION_DIR = Path.home() / ".volition"
AUDIT_LOG = VOLITION_DIR / "audit.log"


# ---------------------------------------------------------------------------
# Audit logging (Constitution Rule 7: log before acting)
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    """Create ~/.volition if needed."""
    VOLITION_DIR.mkdir(parents=True, exist_ok=True)


def _audit_plan(plan: ActionPlan, status: str = "started") -> None:
    """Write plan to audit log."""
    _ensure_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan.plan_id,
        "status": status,
        "action": plan.original_action,
        "steps": [
            {"step_id": s.step_id, "handler": s.handler, "action": s.action}
            for s in plan.steps
        ],
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _audit_outcome(outcome: ActionOutcome) -> None:
    """Write step outcome to audit log."""
    _ensure_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_id": outcome.plan_id,
        "step_id": outcome.step_id,
        "handler": outcome.handler,
        "status": outcome.status,
        "duration_ms": outcome.duration_ms,
        "fallbacks_attempted": outcome.fallbacks_attempted,
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------

def _emit_event(event_type_name: str, payload: dict[str, Any]) -> None:
    """Emit an event via shared.events (best-effort)."""
    try:
        from shared.events import EventType, create_event

        event_type = EventType(event_type_name)
        create_event(
            event_type=event_type,
            source_skill="volition",
            payload=payload,
        )
    except ImportError:
        pass
    except Exception:
        logger.debug("Event emission failed: %s", event_type_name, exc_info=True)


def _record_feedback(
    handler: str,
    status: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Record implicit feedback signal (best-effort).

    Success → useful, error → not_useful, fallback → partial.
    """
    try:
        from shared.feedback import FeedbackSignal, SignalType, FeedbackCollector

        signal_map = {
            "success": SignalType.USEFUL,
            "error": SignalType.NOT_USEFUL,
            "fallback_used": SignalType.PARTIAL,
        }
        signal_type = signal_map.get(status, SignalType.PARTIAL)

        signal = FeedbackSignal(
            event_id=uuid4(),
            signal_type=signal_type,
            source=handler,
            context=context or {},
        )
        collector = FeedbackCollector.get_instance()
        # Synchronous path: collector.record() is async (needs file I/O + event bus),
        # but we only need the in-memory score update here. _update_scores is the
        # only synchronous entry point; if FeedbackCollector adds record_sync(),
        # switch to that.
        collector._update_scores(signal)
    except ImportError:
        pass
    except Exception:
        logger.debug("Feedback recording failed for handler '%s'", handler, exc_info=True)


def _log_to_inland_empire(plan: ActionPlan, outcomes: list[ActionOutcome]) -> None:
    """Log completed action to Inland Empire via synergy bus (fire-and-forget)."""
    try:
        from shared.synergies import volition_log_action

        success_count = sum(1 for o in outcomes if o.status == "success")
        volition_log_action(
            action_type="plan_execution",
            description=plan.original_action,
            metadata={
                "plan_id": plan.plan_id,
                "steps": len(plan.steps),
                "successes": success_count,
                "failures": len(outcomes) - success_count,
            },
        )
    except ImportError:
        pass
    except Exception:
        logger.debug("Inland Empire logging failed", exc_info=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def _dispatch_handler(
    handler_name: str,
    action: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch to a registered handler function.

    Returns a dict with at minimum {"status": "success"|"error", ...}.
    """
    handler_config = get_handler(handler_name)
    if handler_config is None or handler_config.dispatch_fn is None:
        return {
            "status": "error",
            "message": f"Handler '{handler_name}' not registered or has no dispatch function",
        }

    try:
        return await handler_config.dispatch_fn(action, inputs)
    except Exception as e:
        logger.error("Handler '%s' raised: %s", handler_name, e)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------

async def execute_plan(plan: ActionPlan) -> list[ActionOutcome]:
    """Execute a validated plan step-by-step in topological order.

    Constitution Rule 6: Abort on step failure (after fallback chain).
    Constitution Rule 7: Audit log written before execution begins.
    """
    # Rule 7: Log before acting
    _audit_plan(plan, status="started")

    # Emit action.started
    _emit_event("action.started", {
        "plan_id": plan.plan_id,
        "action": plan.original_action,
        "step_count": len(plan.steps),
    })

    ordered_steps = topological_sort(plan.steps)
    outcomes: list[ActionOutcome] = []
    step_outputs: dict[str, Any] = {}

    for step in ordered_steps:
        t0 = time.perf_counter()

        # Resolve input references
        resolved = resolve_inputs(step.inputs, step_outputs)

        # Attempt primary handler
        result = await _dispatch_handler(step.handler, step.action, resolved)

        # Fallback chain on failure
        attempted_fallbacks: list[str] = []
        if result.get("status") == "error" and step.fallback_chain:
            for fallback_name in step.fallback_chain:
                attempted_fallbacks.append(fallback_name)
                result = await _dispatch_handler(fallback_name, step.action, resolved)
                if result.get("status") == "success":
                    break

        duration_ms = int((time.perf_counter() - t0) * 1000)
        is_success = result.get("status") == "success"

        outcome = ActionOutcome(
            plan_id=plan.plan_id,
            step_id=step.step_id,
            handler=step.handler,
            status="success" if is_success else "error",
            duration_ms=duration_ms,
            output_summary=result.get("summary", result.get("message", ""))[:500],
            fallbacks_attempted=attempted_fallbacks,
            data=result,
        )
        outcomes.append(outcome)

        # Audit the outcome
        _audit_outcome(outcome)

        # Record implicit feedback
        if attempted_fallbacks and is_success:
            feedback_status = "fallback_used"
        elif is_success:
            feedback_status = "success"
        else:
            feedback_status = "error"
        _record_feedback(step.handler, feedback_status, {
            "plan_id": plan.plan_id,
            "step_id": step.step_id,
            "action": step.action,
        })
        outcome.feedback_recorded = True

        # Store output for downstream steps
        step_outputs[step.step_id] = result

        # Constitution Rule 6: Abort on failure
        if not is_success:
            _emit_event("action.failed", {
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "handler": step.handler,
                "error": result.get("message", "Unknown error"),
            })
            break

    # Completion event
    all_ok = all(o.status == "success" for o in outcomes)
    if all_ok:
        _emit_event("action.completed", {
            "plan_id": plan.plan_id,
            "action": plan.original_action,
            "outcomes": len(outcomes),
        })

    # Final audit entry
    _audit_plan(plan, status="completed" if all_ok else "failed")

    # Log to Inland Empire
    _log_to_inland_empire(plan, outcomes)

    return outcomes

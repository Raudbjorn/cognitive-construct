"""RhetoricEngine — the pipeline orchestrator.

This is the main entry point for the Rhetoric skill. It coordinates:
1. Decomposition (intent → Toulmin graph)
2. Validation (graph → flags/valid/invalid)
3. Remediation (flags → adjusted graph, max 2 iterations)
4. Strategy selection (audience model → delivery ordering)
5. Analogy bridge (optional, validated graph → structural analogy)

The engine does NOT generate text. It produces a validated, ordered
argument structure that the base model then follows during generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from toulmin.bridge import generate_analogy
from toulmin.decompose import DecompositionError, decompose
from toulmin.models import (
    Analogy,
    ArgumentGraph,
    AudienceModel,
    Confidence,
    DeliveryStrategy,
    FlagSeverity,
    ValidationResult,
    ValidationStatus,
)
from toulmin.strategy import describe_strategy, select_strategy
from toulmin.validate import validate

logger = logging.getLogger(__name__)

MAX_REMEDIATION_ITERATIONS = 2


@dataclass(frozen=True)
class RhetoricPlan:
    """The complete output of the Rhetoric engine — a validated argument plan.

    This is what the base model receives as a generation constraint.
    """

    graph: ArgumentGraph
    validation: ValidationResult
    strategy: DeliveryStrategy
    strategy_description: str
    analogy: Analogy | None = None
    iterations: int = 1

    @property
    def is_sound(self) -> bool:
        return self.validation.status in (ValidationStatus.VALID, ValidationStatus.FLAGGED)

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines = [
            f"Argument: {self.graph.claim.text}",
            f"Scope: {self.graph.claim.scope.value} | Confidence: {self.graph.claim.confidence.value}",
            f"Evidence: {len(self.graph.data)} pieces | Rebuttals: {len(self.graph.rebuttals)}",
            f"Inference: {self.graph.warrant.inference_type.value}",
            f"Validation: {self.validation.status.value} ({len(self.validation.flags)} flags, {len(self.validation.critical_flags)} critical)",
            f"Strategy: {self.strategy.value}",
            f"Iterations: {self.iterations}",
        ]

        if self.validation.qualifier_calibration:
            cal = self.validation.qualifier_calibration
            lines.append(f"Qualifier: {cal.current.value} → {cal.recommended.value} ({'calibrated' if cal.calibrated else 'NEEDS ADJUSTMENT'})")

        if self.analogy:
            lines.append(f"Analogy: {self.analogy.source_domain} → {self.analogy.target_domain} ({'structural' if self.analogy.structural_match else 'SURFACE ONLY'})")

        if self.validation.flags:
            lines.append("\nFlags:")
            for f in self.validation.flags:
                marker = "🔴" if f.severity == FlagSeverity.CRITICAL else "🟡" if f.severity == FlagSeverity.WARNING else "🔵"
                lines.append(f"  {marker} [{f.type}] {f.description}")
                lines.append(f"    → Fix: {f.remediation}")

        return "\n".join(lines)


@dataclass(frozen=True)
class EngineError:
    """Pipeline-level failure."""

    stage: str
    message: str
    details: str | None = None


@dataclass
class RhetoricEngine:
    """Orchestrates the full rhetoric pipeline.

    Attributes:
        api_base: Mercury API endpoint.
        api_key: Bearer token for Mercury.
        model: Mercury model name.
        audience: The current conversation's audience model.
        enable_bridge: Whether to invoke the analogy bridge.
    """

    api_base: str = "https://api.inceptionlabs.ai"
    api_key: str = ""
    model: str = "mercury-2"
    audience: AudienceModel = field(default_factory=AudienceModel)
    enable_bridge: bool = True

    async def plan(
        self,
        intent: str,
        *,
        known_contradictions: list[str] | None = None,
        known_support: list[str] | None = None,
    ) -> RhetoricPlan | EngineError:
        """Run the full rhetoric pipeline on an argument intent.

        Args:
            intent: Natural language description of the argument to make.
            known_contradictions: Evidence from Encyclopedia that contradicts the claim.
            known_support: Evidence from Encyclopedia that supports the claim.

        Returns:
            RhetoricPlan on success, EngineError on failure.
        """
        # Step 1: Decompose intent into Toulmin graph
        logger.info("Step 1: Decomposing intent into Toulmin graph")
        result = await decompose(
            intent,
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
        )

        if isinstance(result, DecompositionError):
            return EngineError(
                stage="decomposition",
                message=result.message,
                details=result.raw_response,
            )

        graph = result

        # Step 2-3: Validate with remediation loop
        logger.info("Step 2: Validating argument graph")
        validation = validate(
            graph,
            known_contradictions=known_contradictions,
            known_support=known_support,
        )

        iterations = 1

        while (
            validation.status == ValidationStatus.FLAGGED
            and iterations < MAX_REMEDIATION_ITERATIONS
        ):
            logger.info("Step 2.%d: Attempting remediation (iteration %d)", iterations, iterations + 1)
            graph = _apply_remediations(graph, validation)
            validation = validate(
                graph,
                known_contradictions=known_contradictions,
                known_support=known_support,
            )
            iterations += 1

        # Step 4: Select delivery strategy
        logger.info("Step 3: Selecting delivery strategy")
        strategy = select_strategy(self.audience)
        strategy_desc = describe_strategy(strategy)

        # Step 5: Analogy bridge (optional, only for valid/flagged arguments)
        analogy: Analogy | None = None
        if (
            self.enable_bridge
            and self.api_key
            and validation.status != ValidationStatus.INVALID
        ):
            logger.info("Step 4: Generating structural analogy via Mercury")
            bridge_result = await generate_analogy(
                graph,
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
            )
            if isinstance(bridge_result, Analogy):
                if bridge_result.structural_match:
                    analogy = bridge_result
                else:
                    logger.info("Analogy discarded: surface similarity only")
            else:
                logger.warning("Analogy bridge failed: %s", bridge_result.message)

        plan = RhetoricPlan(
            graph=graph,
            validation=validation,
            strategy=strategy,
            strategy_description=strategy_desc,
            analogy=analogy,
            iterations=iterations,
        )

        logger.info(
            "Rhetoric plan complete: status=%s strategy=%s iterations=%d",
            validation.status.value,
            strategy.value,
            iterations,
        )

        return plan

    async def validate_only(
        self,
        graph: ArgumentGraph,
        *,
        known_contradictions: list[str] | None = None,
        known_support: list[str] | None = None,
    ) -> ValidationResult:
        """Run validation without decomposition — for pre-built graphs.

        Useful for testing and for arguments constructed manually.
        """
        return validate(
            graph,
            known_contradictions=known_contradictions,
            known_support=known_support,
        )


def _apply_remediations(graph: ArgumentGraph, result: ValidationResult) -> ArgumentGraph:
    """Apply mechanical remediations to a flagged graph.

    This is deliberately conservative — it only applies changes that are
    unambiguously correct. Anything requiring judgment is left as a flag
    for the agent to handle.

    Returns a new graph (immutable update pattern).
    """
    # We operate on a copy to maintain immutability
    updated = graph.model_copy(deep=True)

    for flag in result.flags:
        # Qualifier calibration — the engine can fix this mechanically
        if flag.type == "overclaim" and result.qualifier_calibration:
            if updated.qualifier:
                updated.qualifier = updated.qualifier.model_copy(update={
                    "strength": result.qualifier_calibration.recommended,
                    "text": _qualifier_text(result.qualifier_calibration.recommended),
                    "calibrated": True,
                })
            # Also adjust claim confidence to match
            updated.claim = updated.claim.model_copy(update={
                "confidence": result.qualifier_calibration.recommended,
            })

    return updated


def _qualifier_text(confidence: Confidence) -> str:
    """Generate appropriate hedging language for a confidence level."""
    return {
        Confidence.CERTAIN: "necessarily",
        Confidence.PROBABLE: "in most cases",
        Confidence.PLAUSIBLE: "it is plausible that",
        Confidence.POSSIBLE: "it is possible that",
        Confidence.SPECULATIVE: "speculatively",
    }.get(confidence, "possibly")

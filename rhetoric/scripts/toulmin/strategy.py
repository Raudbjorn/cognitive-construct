"""Delivery strategy selection based on audience epistemic state.

The strategy doesn't change WHAT the argument says — it changes the ORDER
in which nodes are presented. Same graph, different paths through it.
"""

from __future__ import annotations

from toulmin.models import (
    AudienceModel,
    AudiencePosition,
    DeliveryStrategy,
    KnowledgeLevel,
)


def select_strategy(audience: AudienceModel) -> DeliveryStrategy:
    """Select the optimal delivery strategy for the current audience state.

    Pure function: same audience model → same strategy.
    """
    position = audience.epistemic.position
    knowledge = audience.epistemic.knowledge_level
    pushback_ratio = _pushback_ratio(audience)

    # Opposed or highly skeptical → lead with their objections, then refute
    if position == AudiencePosition.OPPOSED:
        return DeliveryStrategy.PROLEPTIC

    if position == AudiencePosition.SKEPTICAL:
        # High pushback rate amplifies skepticism
        if pushback_ratio > 0.5:
            return DeliveryStrategy.PROLEPTIC
        return DeliveryStrategy.CONCESSIVE

    # Novice → scaffold prerequisite knowledge first
    if knowledge == KnowledgeLevel.NOVICE:
        return DeliveryStrategy.SCAFFOLDED

    # Unknown → ask questions to establish common ground
    if position == AudiencePosition.UNKNOWN and knowledge == KnowledgeLevel.UNKNOWN:
        # Could go either way. Default to direct (least presumptuous)
        # per the spec's cold-start decision.
        return DeliveryStrategy.DIRECT

    # Neutral or agrees → direct assertion is efficient
    return DeliveryStrategy.DIRECT


def _pushback_ratio(audience: AudienceModel) -> float:
    """Ratio of pushbacks to total interactions."""
    total = audience.signals.pushbacks + audience.signals.agreements + audience.signals.questions_asked
    if total == 0:
        return 0.0
    return audience.signals.pushbacks / total


def describe_strategy(strategy: DeliveryStrategy) -> str:
    """Human-readable description of how the argument should be ordered."""
    descriptions = {
        DeliveryStrategy.DIRECT: (
            "Lead with the claim, then present evidence and warrant. "
            "Efficient when audience is receptive."
        ),
        DeliveryStrategy.PROLEPTIC: (
            "Lead with the strongest counterargument and its refutation, "
            "THEN present evidence and claim. Disarms skepticism before it forms."
        ),
        DeliveryStrategy.SOCRATIC: (
            "Lead with questions that reveal prerequisite gaps, then scaffold "
            "understanding before presenting the argument."
        ),
        DeliveryStrategy.SCAFFOLDED: (
            "Lead with background context and definitions, building up to the "
            "evidence and claim. For audiences who lack prerequisite knowledge."
        ),
        DeliveryStrategy.CONCESSIVE: (
            "Acknowledge the valid parts of the audience's objection, then pivot "
            "to evidence that complicates their view. Builds trust through honesty."
        ),
    }
    return descriptions.get(strategy, "Unknown strategy")

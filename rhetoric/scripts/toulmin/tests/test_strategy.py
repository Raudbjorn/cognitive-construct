"""Tests for audience-based delivery strategy selection."""

from __future__ import annotations

from toulmin.models import (
    AudienceModel,
    AudiencePosition,
    ConversationSignals,
    DeliveryStrategy,
    EpistemicState,
    KnowledgeLevel,
)
from toulmin.strategy import select_strategy


class TestStrategySelection:
    def test_opposed_audience_gets_proleptic(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(position=AudiencePosition.OPPOSED),
        )
        assert select_strategy(audience) == DeliveryStrategy.PROLEPTIC

    def test_skeptical_with_high_pushback_gets_proleptic(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(position=AudiencePosition.SKEPTICAL),
            signals=ConversationSignals(pushbacks=5, agreements=1),
        )
        assert select_strategy(audience) == DeliveryStrategy.PROLEPTIC

    def test_skeptical_with_low_pushback_gets_concessive(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(position=AudiencePosition.SKEPTICAL),
            signals=ConversationSignals(pushbacks=1, agreements=5),
        )
        assert select_strategy(audience) == DeliveryStrategy.CONCESSIVE

    def test_novice_gets_scaffolded(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(
                position=AudiencePosition.NEUTRAL,
                knowledge_level=KnowledgeLevel.NOVICE,
            ),
        )
        assert select_strategy(audience) == DeliveryStrategy.SCAFFOLDED

    def test_neutral_expert_gets_direct(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(
                position=AudiencePosition.NEUTRAL,
                knowledge_level=KnowledgeLevel.EXPERT,
            ),
        )
        assert select_strategy(audience) == DeliveryStrategy.DIRECT

    def test_unknown_audience_defaults_to_direct(self) -> None:
        """Cold start: no signal → least presumptuous strategy."""
        audience = AudienceModel()
        assert select_strategy(audience) == DeliveryStrategy.DIRECT

    def test_agreeing_audience_gets_direct(self) -> None:
        audience = AudienceModel(
            epistemic=EpistemicState(position=AudiencePosition.AGREES),
        )
        assert select_strategy(audience) == DeliveryStrategy.DIRECT

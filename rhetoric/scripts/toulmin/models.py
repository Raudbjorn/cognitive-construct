"""Core data models for the Rhetoric engine.

All domain types live here. The Toulmin argument graph is the central structure.
Validation results, audience models, and delivery strategies are typed as enums
and Pydantic models so the symbolic engine operates on well-defined structures,
not free-form strings.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums — constrained vocabularies for the symbolic engine
# ---------------------------------------------------------------------------


class ClaimScope(StrEnum):
    """How broadly the claim applies."""

    UNIVERSAL = "universal"  # "All X are Y"
    GENERAL = "general"  # "Most X are Y"
    PARTICULAR = "particular"  # "Some X are Y"
    EXISTENTIAL = "existential"  # "There exists an X that is Y"


class Confidence(StrEnum):
    """Epistemic strength of the claim."""

    CERTAIN = "certain"
    PROBABLE = "probable"
    PLAUSIBLE = "plausible"
    POSSIBLE = "possible"
    SPECULATIVE = "speculative"

    @property
    def rank(self) -> int:
        """Ordinal rank for comparison. Higher = stronger."""
        return {
            "certain": 5,
            "probable": 4,
            "plausible": 3,
            "possible": 2,
            "speculative": 1,
        }[self.value]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank >= other.rank

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self.rank <= other.rank


class InferenceType(StrEnum):
    """Type of inferential step the warrant represents."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"


class DataSource(StrEnum):
    ENCYCLOPEDIA = "encyclopedia"
    USER_STATED = "user_stated"
    COMMON_KNOWLEDGE = "common_knowledge"
    INFERRED = "inferred"


class DataStrength(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ANECDOTAL = "anecdotal"

    @property
    def rank(self) -> int:
        return {"strong": 4, "moderate": 3, "weak": 2, "anecdotal": 1}[self.value]


class BackingSource(StrEnum):
    ENCYCLOPEDIA = "encyclopedia"
    DOMAIN_EXPERTISE = "domain_expertise"
    EMPIRICAL_RESEARCH = "empirical_research"
    AUTHORITY = "authority"


class RebuttalSeverity(StrEnum):
    CRITICAL = "critical"
    SIGNIFICANT = "significant"
    MINOR = "minor"


class FlagSeverity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class FlagCategory(StrEnum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    STRUCTURAL = "structural"
    CALIBRATION = "calibration"
    INTEGRITY = "integrity"


class ValidationStatus(StrEnum):
    PENDING = "pending"
    VALID = "valid"
    FLAGGED = "flagged"
    INVALID = "invalid"


class DeliveryStrategy(StrEnum):
    """Argument node ordering strategy based on audience state."""

    DIRECT = "direct"  # Claim → Data → Warrant → Qualifier
    PROLEPTIC = "proleptic"  # Rebuttal → Response → Data → Claim
    SOCRATIC = "socratic"  # Questions → Scaffolding → Data → Claim
    SCAFFOLDED = "scaffolded"  # Background → Definitions → Data → Claim
    CONCESSIVE = "concessive"  # Concession → Pivot → Data → Claim


class AudiencePosition(StrEnum):
    AGREES = "agrees"
    NEUTRAL = "neutral"
    SKEPTICAL = "skeptical"
    OPPOSED = "opposed"
    UNKNOWN = "unknown"


class KnowledgeLevel(StrEnum):
    EXPERT = "expert"
    INFORMED = "informed"
    NOVICE = "novice"
    UNKNOWN = "unknown"


class ReasoningStyle(StrEnum):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    EVIDENCE_DRIVEN = "evidence_driven"
    AUTHORITY_DRIVEN = "authority_driven"


# ---------------------------------------------------------------------------
# Toulmin graph nodes
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """The conclusion being argued."""

    text: str
    scope: ClaimScope
    confidence: Confidence


class Datum(BaseModel):
    """A piece of evidence supporting the claim."""

    id: str = Field(default_factory=lambda: f"data-{uuid.uuid4().hex[:6]}")
    text: str
    source: DataSource
    source_ref: str | None = None
    strength: DataStrength


class Warrant(BaseModel):
    """The bridge connecting data to claim — WHY the evidence supports the conclusion."""

    text: str
    inference_type: InferenceType
    explicit: bool = True
    formalization: str | None = None


class Backing(BaseModel):
    """Support for the warrant itself."""

    text: str
    source: BackingSource
    source_ref: str | None = None


class Qualifier(BaseModel):
    """Strength marker / hedge for the claim."""

    text: str
    strength: Confidence
    calibrated: bool = False


class Rebuttal(BaseModel):
    """A counterargument or exception."""

    id: str = Field(default_factory=lambda: f"rebuttal-{uuid.uuid4().hex[:6]}")
    text: str
    severity: RebuttalSeverity
    addressed: bool = False
    response: str | None = None


# ---------------------------------------------------------------------------
# The Toulmin argument graph
# ---------------------------------------------------------------------------


def _make_graph_id() -> str:
    return f"arg-{uuid.uuid4().hex[:8]}"


class ArgumentGraph(BaseModel):
    """A complete Toulmin argument structure.

    This is the central data structure of the Rhetoric engine.
    Every argument is decomposed into this graph before validation.
    """

    id: str = Field(default_factory=_make_graph_id)
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))

    claim: Claim
    data: Annotated[list[Datum], Field(min_length=1)]
    warrant: Warrant
    backing: Backing | None = None
    qualifier: Qualifier | None = None
    rebuttals: list[Rebuttal] = Field(default_factory=list)

    sub_arguments: list[str] = Field(
        default_factory=list,
        description="IDs of nested ArgumentGraphs for complex claims",
    )

    validation: ValidationStatus = ValidationStatus.PENDING


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


class Flag(BaseModel):
    """A specific issue found during validation."""

    type: str
    severity: FlagSeverity
    location: str  # node ID or field name
    description: str
    remediation: str
    category: FlagCategory


class QualifierCalibration(BaseModel):
    current: Confidence
    recommended: Confidence
    calibrated: bool = False


class StructuralCompleteness(BaseModel):
    has_claim: bool = False
    has_data: bool = False
    has_warrant: bool = False
    has_backing: bool = False
    has_qualifier: bool = False
    has_rebuttals: bool = False
    missing: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Output of the validation engine for a single argument graph."""

    graph_id: str
    status: ValidationStatus
    structural: StructuralCompleteness
    flags: list[Flag] = Field(default_factory=list)
    qualifier_calibration: QualifierCalibration | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID

    @property
    def critical_flags(self) -> list[Flag]:
        return [f for f in self.flags if f.severity == FlagSeverity.CRITICAL]


# ---------------------------------------------------------------------------
# Audience model
# ---------------------------------------------------------------------------


class ResistancePoint(BaseModel):
    topic: str
    strength: str  # "strong" | "moderate" | "mild"
    basis: str


class EpistemicState(BaseModel):
    position: AudiencePosition = AudiencePosition.UNKNOWN
    knowledge_level: KnowledgeLevel = KnowledgeLevel.UNKNOWN
    reasoning_style: ReasoningStyle = ReasoningStyle.ANALYTICAL
    resistance_points: list[ResistancePoint] = Field(default_factory=list)


class ConversationSignals(BaseModel):
    questions_asked: int = 0
    pushbacks: int = 0
    agreements: int = 0
    topic_changes: int = 0


class AudienceModel(BaseModel):
    """Tracks the listener's epistemic state across a conversation."""

    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    epistemic: EpistemicState = Field(default_factory=EpistemicState)
    signals: ConversationSignals = Field(default_factory=ConversationSignals)
    recommended_strategy: DeliveryStrategy = DeliveryStrategy.DIRECT


# ---------------------------------------------------------------------------
# Analogy bridge output
# ---------------------------------------------------------------------------


class Analogy(BaseModel):
    """A structurally-constrained analogy from the diffusion bridge."""

    source_domain: str
    target_domain: str
    mapping: dict[str, str]  # source element → target element
    analogy_text: str
    structural_match: bool = False  # set by verification

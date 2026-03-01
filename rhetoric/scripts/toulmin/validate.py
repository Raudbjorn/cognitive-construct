"""Four-pass symbolic validation engine for Toulmin argument graphs.

This is the irreducible core of the Rhetoric skill — the thing an LLM
cannot do internally. Pure functions, no LLM calls, deterministic output.

Pass 1: Structural completeness (are all required nodes present?)
Pass 2: Inferential type validation (is each reasoning step sound?)
Pass 3: Cross-reference integrity (does the argument ignore known evidence?)
Pass 4: Qualifier calibration (does claim strength match evidence strength?)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from toulmin.models import (
    Confidence,
    DataStrength,
    Flag,
    FlagCategory,
    FlagSeverity,
    InferenceType,
    QualifierCalibration,
    StructuralCompleteness,
    ValidationResult,
    ValidationStatus,
)

if TYPE_CHECKING:
    from toulmin.models import ArgumentGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known deductive forms (pattern matching on formalization strings)
# ---------------------------------------------------------------------------

# These regex patterns match common logical form annotations.
# The formalization field is free-text, so we look for structural markers.

_VALID_DEDUCTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Match both orderings: "A, A→B ⊢ B" and "A→B, A ⊢ B"
    ("modus_ponens", re.compile(r"A.*,\s*A\s*→\s*B\s*⊢\s*B|A\s*→\s*B.*,\s*A\s*⊢\s*B", re.IGNORECASE)),
    ("modus_tollens", re.compile(r"¬B.*,\s*A\s*→\s*B\s*⊢\s*¬A|A\s*→\s*B.*,\s*¬B\s*⊢\s*¬A", re.IGNORECASE)),
    ("hypothetical_syllogism", re.compile(r"A\s*→\s*B.*,\s*B\s*→\s*C\s*⊢\s*A\s*→\s*C", re.IGNORECASE)),
    ("disjunctive_syllogism", re.compile(r"A\s*∨\s*B.*,\s*¬A\s*⊢\s*B", re.IGNORECASE)),
]

_INVALID_DEDUCTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("affirming_consequent", re.compile(r"B.*,\s*A.*→.*B\s*⊢\s*A", re.IGNORECASE)),
    ("denying_antecedent", re.compile(r"¬A.*,\s*A.*→.*B\s*⊢\s*¬B", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Inductive heuristics
# ---------------------------------------------------------------------------

# Keywords that suggest universal/unhedged claims
_UNIVERSAL_MARKERS = re.compile(
    r"\b(all|every|always|never|none|no one|everyone|everything)\b",
    re.IGNORECASE,
)

# Keywords that suggest anecdotal evidence
_ANECDOTAL_MARKERS = re.compile(
    r"\b(blog post|tweet|anecdote|one person|I heard|someone said|a friend)\b",
    re.IGNORECASE,
)

# Keywords suggesting temporal-causal confusion
_TEMPORAL_CAUSAL_MARKERS = re.compile(
    r"\b(after|since|following|then|subsequently|led to|caused)\b",
    re.IGNORECASE,
)

# Sample size extraction (crude but functional for PoC)
_SAMPLE_SIZE_PATTERN = re.compile(r"n\s*=\s*(\d+)|(\d+)\s*(users?|people|respondents|surveyed)")


def _extract_sample_size(text: str) -> int | None:
    """Try to extract a numeric sample size from evidence text."""
    match = _SAMPLE_SIZE_PATTERN.search(text)
    if not match:
        return None
    # n=X form or "X users" form
    return int(match.group(1) or match.group(2))


# ---------------------------------------------------------------------------
# Evidence strength → recommended confidence mapping
# ---------------------------------------------------------------------------

_EVIDENCE_CONFIDENCE_MAP: dict[tuple[InferenceType, DataStrength], Confidence] = {
    # Deductive — if valid, strength comes from form, not data
    (InferenceType.DEDUCTIVE, DataStrength.STRONG): Confidence.CERTAIN,
    (InferenceType.DEDUCTIVE, DataStrength.MODERATE): Confidence.CERTAIN,
    (InferenceType.DEDUCTIVE, DataStrength.WEAK): Confidence.PROBABLE,
    (InferenceType.DEDUCTIVE, DataStrength.ANECDOTAL): Confidence.PLAUSIBLE,
    # Inductive
    (InferenceType.INDUCTIVE, DataStrength.STRONG): Confidence.PROBABLE,
    (InferenceType.INDUCTIVE, DataStrength.MODERATE): Confidence.PLAUSIBLE,
    (InferenceType.INDUCTIVE, DataStrength.WEAK): Confidence.POSSIBLE,
    (InferenceType.INDUCTIVE, DataStrength.ANECDOTAL): Confidence.SPECULATIVE,
    # Abductive
    (InferenceType.ABDUCTIVE, DataStrength.STRONG): Confidence.PLAUSIBLE,
    (InferenceType.ABDUCTIVE, DataStrength.MODERATE): Confidence.PLAUSIBLE,
    (InferenceType.ABDUCTIVE, DataStrength.WEAK): Confidence.POSSIBLE,
    (InferenceType.ABDUCTIVE, DataStrength.ANECDOTAL): Confidence.SPECULATIVE,
    # Analogical — inherently weaker
    (InferenceType.ANALOGICAL, DataStrength.STRONG): Confidence.PLAUSIBLE,
    (InferenceType.ANALOGICAL, DataStrength.MODERATE): Confidence.POSSIBLE,
    (InferenceType.ANALOGICAL, DataStrength.WEAK): Confidence.SPECULATIVE,
    (InferenceType.ANALOGICAL, DataStrength.ANECDOTAL): Confidence.SPECULATIVE,
}


def _recommended_confidence(inference_type: InferenceType, evidence: DataStrength) -> Confidence:
    """Map (inference_type, weakest_evidence) → maximum justified confidence."""
    return _EVIDENCE_CONFIDENCE_MAP.get(
        (inference_type, evidence),
        Confidence.SPECULATIVE,  # fallback: assume weakest
    )


# ---------------------------------------------------------------------------
# Pass 1: Structural completeness
# ---------------------------------------------------------------------------


def _pass_structural(graph: ArgumentGraph) -> tuple[StructuralCompleteness, list[Flag]]:
    """Check that all required and recommended nodes exist."""
    flags: list[Flag] = []
    sc = StructuralCompleteness()

    # Required nodes
    sc.has_claim = bool(graph.claim and graph.claim.text.strip())
    sc.has_data = bool(graph.data and all(d.text.strip() for d in graph.data))
    sc.has_warrant = bool(graph.warrant and graph.warrant.text.strip())

    for field, present in [("claim", sc.has_claim), ("data", sc.has_data), ("warrant", sc.has_warrant)]:
        if not present:
            sc.missing.append(field)
            flags.append(Flag(
                type="missing_required",
                severity=FlagSeverity.CRITICAL,
                location=field,
                description=f"Required node '{field}' is missing or empty",
                remediation=f"Provide a valid {field} before the argument can be validated",
                category=FlagCategory.STRUCTURAL,
            ))

    # Recommended nodes
    sc.has_backing = graph.backing is not None and bool(graph.backing.text.strip())
    sc.has_qualifier = graph.qualifier is not None and bool(graph.qualifier.text.strip())
    sc.has_rebuttals = bool(graph.rebuttals)

    if not sc.has_backing:
        sc.missing.append("backing")
        flags.append(Flag(
            type="missing_backing",
            severity=FlagSeverity.WARNING,
            location="backing",
            description="Warrant has no backing — the bridge is unsupported",
            remediation="Add evidence or authority that supports the warrant itself",
            category=FlagCategory.STRUCTURAL,
        ))

    if not sc.has_qualifier:
        sc.missing.append("qualifier")
        flags.append(Flag(
            type="missing_qualifier",
            severity=FlagSeverity.WARNING,
            location="qualifier",
            description="Claim has no qualifier — strength is unstated",
            remediation="Add a qualifier indicating how strongly the claim should be stated",
            category=FlagCategory.STRUCTURAL,
        ))

    if not sc.has_rebuttals:
        sc.missing.append("rebuttals")
        flags.append(Flag(
            type="missing_rebuttals",
            severity=FlagSeverity.WARNING,
            location="rebuttals",
            description="No counterarguments considered",
            remediation="Address at least one potential objection to strengthen the argument",
            category=FlagCategory.STRUCTURAL,
        ))

    # Implicit warrant is higher risk
    if sc.has_warrant and not graph.warrant.explicit:
        flags.append(Flag(
            type="implicit_warrant",
            severity=FlagSeverity.WARNING,
            location="warrant",
            description="Warrant is implicit (reconstructed, not stated) — higher risk of invalid bridge",
            remediation="Consider making the warrant explicit to reduce ambiguity",
            category=FlagCategory.STRUCTURAL,
        ))

    return sc, flags


# ---------------------------------------------------------------------------
# Pass 2: Inferential type validation
# ---------------------------------------------------------------------------


def _pass_inferential(graph: ArgumentGraph) -> list[Flag]:
    """Validate the warrant's inferential step based on its type."""
    flags: list[Flag] = []
    w = graph.warrant

    if w.inference_type == InferenceType.DEDUCTIVE:
        flags.extend(_validate_deductive(graph))
    elif w.inference_type == InferenceType.INDUCTIVE:
        flags.extend(_validate_inductive(graph))
    elif w.inference_type == InferenceType.ABDUCTIVE:
        flags.extend(_validate_abductive(graph))
    elif w.inference_type == InferenceType.ANALOGICAL:
        flags.extend(_validate_analogical(graph))

    return flags


def _validate_deductive(graph: ArgumentGraph) -> list[Flag]:
    """Check deductive arguments against known valid/invalid syllogistic forms."""
    flags: list[Flag] = []
    formalization = graph.warrant.formalization or ""

    if not formalization.strip():
        flags.append(Flag(
            type="missing_formalization",
            severity=FlagSeverity.WARNING,
            location="warrant.formalization",
            description="Deductive warrant has no logical formalization — cannot verify form",
            remediation="Provide a logical form (e.g. 'A, A→B ⊢ B') for mechanical verification",
            category=FlagCategory.DEDUCTIVE,
        ))
        return flags

    # Check invalid forms first (they're more dangerous to miss)
    for name, pattern in _INVALID_DEDUCTIVE_PATTERNS:
        if pattern.search(formalization):
            flags.append(Flag(
                type=name,
                severity=FlagSeverity.CRITICAL,
                location="warrant",
                description=f"Deductive form matches known fallacy: {name}",
                remediation=f"This is a formal fallacy ({name}). The conclusion does not follow from the premises.",
                category=FlagCategory.DEDUCTIVE,
            ))
            return flags  # early return: one formal fallacy is enough

    # Check valid forms
    for name, pattern in _VALID_DEDUCTIVE_PATTERNS:
        if pattern.search(formalization):
            logger.info("Deductive form validated as %s", name)
            return flags  # valid, no flags

    # No pattern matched
    flags.append(Flag(
        type="unrecognized_deductive_form",
        severity=FlagSeverity.WARNING,
        location="warrant",
        description=f"Deductive form '{formalization}' not recognized — verify manually",
        remediation="Ensure the logical form follows a valid syllogistic pattern",
        category=FlagCategory.DEDUCTIVE,
    ))

    return flags


def _validate_inductive(graph: ArgumentGraph) -> list[Flag]:
    """Check inductive arguments for known failure modes."""
    flags: list[Flag] = []
    claim_text = graph.claim.text.lower()
    claim_scope = graph.claim.scope

    for datum in graph.data:
        text = datum.text

        # Hasty generalization: small sample + broad claim
        sample_size = _extract_sample_size(text)
        if sample_size is not None and sample_size < 30 and claim_scope.value in ("universal", "general"):
            flags.append(Flag(
                type="hasty_generalization",
                severity=FlagSeverity.CRITICAL,
                location=datum.id,
                description=f"Sample size n={sample_size} is too small for {claim_scope.value} claim",
                remediation="Either narrow claim scope to 'particular' or strengthen evidence (n≥30 for general claims)",
                category=FlagCategory.INDUCTIVE,
            ))
        # Also flag when evidence is anecdotal/weak with broad claims, even without
        # an extractable sample size
        elif (
            sample_size is None
            and datum.strength.rank <= DataStrength.WEAK.rank
            and claim_scope.value in ("universal", "general")
        ):
            flags.append(Flag(
                type="hasty_generalization",
                severity=FlagSeverity.CRITICAL,
                location=datum.id,
                description=f"{datum.strength.value.title()} evidence cannot support a {claim_scope.value} claim",
                remediation="Either narrow claim scope to 'particular' or provide systematic evidence",
                category=FlagCategory.INDUCTIVE,
            ))

        # Anecdotal evidence + strong claim
        if datum.strength.value == "anecdotal" and _UNIVERSAL_MARKERS.search(claim_text):
            flags.append(Flag(
                type="anecdotal_overclaim",
                severity=FlagSeverity.CRITICAL,
                location=datum.id,
                description="Universal claim based on anecdotal evidence",
                remediation="Downgrade claim scope or replace anecdotal evidence with systematic data",
                category=FlagCategory.INDUCTIVE,
            ))

        # Check for selection/survivorship bias signals
        if _ANECDOTAL_MARKERS.search(text) and datum.strength.value != "anecdotal":
            flags.append(Flag(
                type="misclassified_evidence",
                severity=FlagSeverity.WARNING,
                location=datum.id,
                description="Evidence text contains anecdotal markers but is classified as stronger",
                remediation=f"Consider reclassifying evidence strength from '{datum.strength.value}' to 'anecdotal'",
                category=FlagCategory.INDUCTIVE,
            ))

    # Temporal-causal confusion in warrant
    warrant_text = graph.warrant.text.lower()
    if _TEMPORAL_CAUSAL_MARKERS.search(warrant_text) and "cause" in claim_text:
        flags.append(Flag(
            type="false_cause",
            severity=FlagSeverity.WARNING,
            location="warrant",
            description="Warrant uses temporal language for a causal claim — post hoc risk",
            remediation="Provide a mechanism or controlled evidence, not just temporal sequence",
            category=FlagCategory.INDUCTIVE,
        ))

    return flags


def _validate_abductive(graph: ArgumentGraph) -> list[Flag]:
    """Check abductive arguments for explanatory completeness."""
    flags: list[Flag] = []

    # Single explanation: no alternative hypotheses considered
    has_alternative_rebuttal = any(
        "alternative" in r.text.lower() or "other explanation" in r.text.lower()
        for r in graph.rebuttals
    )
    if not has_alternative_rebuttal:
        flags.append(Flag(
            type="single_explanation",
            severity=FlagSeverity.WARNING,
            location="rebuttals",
            description="Abductive argument considers no alternative explanations",
            remediation="Address at least one competing hypothesis in rebuttals",
            category=FlagCategory.ABDUCTIVE,
        ))

    return flags


def _validate_analogical(graph: ArgumentGraph) -> list[Flag]:
    """Check analogical arguments for structural vs surface similarity."""
    flags: list[Flag] = []

    # Surface similarity markers in warrant
    surface_markers = re.compile(
        r"\b(looks? like|similar to|reminds? me of|resembles?)\b",
        re.IGNORECASE,
    )
    structural_markers = re.compile(
        r"\b(functions? as|same mechanism|structurally|operates? like|same role)\b",
        re.IGNORECASE,
    )

    warrant_text = graph.warrant.text
    has_surface = bool(surface_markers.search(warrant_text))
    has_structural = bool(structural_markers.search(warrant_text))

    if has_surface and not has_structural:
        flags.append(Flag(
            type="surface_analogy",
            severity=FlagSeverity.WARNING,
            location="warrant",
            description="Analogical warrant references surface similarity without structural mapping",
            remediation="Identify the causal/functional relationship that is shared, not just appearance",
            category=FlagCategory.ANALOGICAL,
        ))

    return flags


# ---------------------------------------------------------------------------
# Pass 3: Cross-reference integrity
# ---------------------------------------------------------------------------


def _pass_crossref(
    graph: ArgumentGraph,
    known_contradictions: list[str] | None = None,
    known_support: list[str] | None = None,
) -> list[Flag]:
    """Check the argument against externally-known evidence.

    In production, this queries the Encyclopedia. For PoC, contradictions and
    supporting evidence are passed in explicitly.
    """
    flags: list[Flag] = []

    for contradiction in (known_contradictions or []):
        # Check if any rebuttal addresses this
        addressed = any(
            contradiction.lower() in r.text.lower() or r.response and contradiction.lower() in r.response.lower()
            for r in graph.rebuttals
        )
        if not addressed:
            flags.append(Flag(
                type="incomplete_rebuttal",
                severity=FlagSeverity.CRITICAL,
                location="rebuttals",
                description=f"Known contradicting evidence not addressed: '{contradiction[:80]}...'",
                remediation="Add a rebuttal that addresses this contradicting evidence",
                category=FlagCategory.INTEGRITY,
            ))

    for support in (known_support or []):
        in_data = any(support.lower() in d.text.lower() for d in graph.data)
        if not in_data:
            flags.append(Flag(
                type="unused_support",
                severity=FlagSeverity.INFO,
                location="data",
                description=f"Additional supporting evidence available: '{support[:80]}...'",
                remediation="Consider including this evidence to strengthen the argument",
                category=FlagCategory.INTEGRITY,
            ))

    return flags


# ---------------------------------------------------------------------------
# Pass 4: Qualifier calibration
# ---------------------------------------------------------------------------


def _pass_calibration(graph: ArgumentGraph) -> tuple[QualifierCalibration | None, list[Flag]]:
    """Ensure claim strength matches evidence strength."""
    flags: list[Flag] = []

    if not graph.qualifier:
        return None, flags

    # Weakest evidence determines the chain's strength
    weakest = min(graph.data, key=lambda d: d.strength.rank)
    recommended = _recommended_confidence(graph.warrant.inference_type, weakest.strength)
    current = graph.qualifier.strength

    cal = QualifierCalibration(
        current=current,
        recommended=recommended,
        calibrated=False,
    )

    if current > recommended:
        gap = current.rank - recommended.rank
        severity = FlagSeverity.CRITICAL if gap >= 2 else FlagSeverity.WARNING
        flags.append(Flag(
            type="overclaim",
            severity=severity,
            location="qualifier",
            description=(
                f"Qualifier '{current.value}' is stronger than evidence supports "
                f"(recommended: '{recommended.value}')"
            ),
            remediation=f"Adjust qualifier from '{current.value}' to '{recommended.value}'",
            category=FlagCategory.CALIBRATION,
        ))
    elif current < recommended and (recommended.rank - current.rank) >= 2:
        flags.append(Flag(
            type="underclaim",
            severity=FlagSeverity.INFO,
            location="qualifier",
            description=(
                f"Qualifier '{current.value}' is weaker than evidence supports "
                f"(evidence justifies '{recommended.value}')"
            ),
            remediation=f"Evidence supports upgrading qualifier to '{recommended.value}'",
            category=FlagCategory.CALIBRATION,
        ))
    else:
        cal.calibrated = True

    return cal, flags


# ---------------------------------------------------------------------------
# The full validation pipeline
# ---------------------------------------------------------------------------


def validate(
    graph: ArgumentGraph,
    *,
    known_contradictions: list[str] | None = None,
    known_support: list[str] | None = None,
) -> ValidationResult:
    """Run all four validation passes on an argument graph.

    Returns a deterministic ValidationResult for the same input.
    This function is pure — no side effects, no LLM calls.
    """
    all_flags: list[Flag] = []

    # Pass 1: Structural completeness
    structural, struct_flags = _pass_structural(graph)
    all_flags.extend(struct_flags)

    # If structurally incomplete (missing required nodes), stop early
    has_critical_structural = any(
        f.severity == FlagSeverity.CRITICAL for f in struct_flags
    )
    if has_critical_structural:
        return ValidationResult(
            graph_id=graph.id,
            status=ValidationStatus.INVALID,
            structural=structural,
            flags=all_flags,
        )

    # Pass 2: Inferential type validation
    inf_flags = _pass_inferential(graph)
    all_flags.extend(inf_flags)

    # Pass 3: Cross-reference integrity
    xref_flags = _pass_crossref(graph, known_contradictions, known_support)
    all_flags.extend(xref_flags)

    # Pass 4: Qualifier calibration
    calibration, cal_flags = _pass_calibration(graph)
    all_flags.extend(cal_flags)

    # Determine overall status
    has_critical = any(f.severity == FlagSeverity.CRITICAL for f in all_flags)
    has_invalid_deductive = any(f.type in ("affirming_consequent", "denying_antecedent") for f in all_flags)
    has_warnings = any(f.severity == FlagSeverity.WARNING for f in all_flags)

    if has_invalid_deductive:
        status = ValidationStatus.INVALID
    elif has_critical or has_warnings:
        status = ValidationStatus.FLAGGED
    else:
        status = ValidationStatus.VALID

    result = ValidationResult(
        graph_id=graph.id,
        status=status,
        structural=structural,
        flags=all_flags,
        qualifier_calibration=calibration,
    )

    logger.info(
        "Validation complete: graph=%s status=%s flags=%d critical=%d",
        graph.id,
        status.value,
        len(all_flags),
        len(result.critical_flags),
    )

    return result

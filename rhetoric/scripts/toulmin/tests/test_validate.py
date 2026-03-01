"""Tests for the symbolic validation engine.

Each fallacy pattern gets a paired test: one valid instance and one invalid
instance with the same surface structure. The test passes only if the engine
correctly distinguishes them.

No LLM calls. Pure deterministic logic.
"""

from __future__ import annotations

from toulmin.models import (
    ArgumentGraph,
    Backing,
    BackingSource,
    Claim,
    ClaimScope,
    Confidence,
    DataSource,
    DataStrength,
    Datum,
    FlagSeverity,
    InferenceType,
    Qualifier,
    Rebuttal,
    RebuttalSeverity,
    ValidationStatus,
    Warrant,
)
from toulmin.validate import validate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinels to distinguish "not provided" from "explicitly None/empty"
_SENTINEL_BACKING = Backing(text="__SENTINEL__", source=BackingSource.AUTHORITY)
_SENTINEL_QUALIFIER = Qualifier(text="__SENTINEL__", strength=Confidence.SPECULATIVE)
_SENTINEL_REBUTTALS: list[Rebuttal] = [Rebuttal(text="__SENTINEL__", severity=RebuttalSeverity.MINOR)]


def _make_graph(
    claim_text: str = "Test claim",
    claim_scope: ClaimScope = ClaimScope.GENERAL,
    claim_confidence: Confidence = Confidence.PROBABLE,
    data: list[Datum] | None = None,
    warrant_text: str = "Test warrant",
    inference_type: InferenceType = InferenceType.INDUCTIVE,
    formalization: str | None = None,
    backing: Backing | None = _SENTINEL_BACKING,
    qualifier: Qualifier | None = _SENTINEL_QUALIFIER,
    rebuttals: list[Rebuttal] | None = _SENTINEL_REBUTTALS,
) -> ArgumentGraph:
    """Helper to build a graph with sensible defaults.

    Pass None explicitly to omit a field. Pass [] for empty rebuttals.
    Default creates a fully-populated graph.
    """
    resolved_backing = (
        Backing(text="Default backing", source=BackingSource.EMPIRICAL_RESEARCH)
        if backing is _SENTINEL_BACKING
        else backing
    )
    resolved_qualifier = (
        Qualifier(text="probably", strength=Confidence.PROBABLE)
        if qualifier is _SENTINEL_QUALIFIER
        else qualifier
    )
    resolved_rebuttals: list[Rebuttal] = (
        [Rebuttal(text="Default rebuttal", severity=RebuttalSeverity.MINOR, addressed=True, response="Addressed")]
        if rebuttals is _SENTINEL_REBUTTALS
        else (rebuttals if rebuttals is not None else [])
    )

    return ArgumentGraph(
        claim=Claim(text=claim_text, scope=claim_scope, confidence=claim_confidence),
        data=data if data is not None else [
            Datum(text="Default evidence", source=DataSource.ENCYCLOPEDIA, strength=DataStrength.STRONG)
        ],
        warrant=Warrant(
            text=warrant_text,
            inference_type=inference_type,
            formalization=formalization,
        ),
        backing=resolved_backing,
        qualifier=resolved_qualifier,
        rebuttals=resolved_rebuttals,
    )


# ---------------------------------------------------------------------------
# Pass 1: Structural completeness
# ---------------------------------------------------------------------------


class TestStructuralCompleteness:
    """Tests for Pass 1: required and recommended nodes."""

    def test_complete_graph_has_no_structural_flags(self) -> None:
        graph = _make_graph()
        result = validate(graph)
        structural_flags = [f for f in result.flags if f.category.value == "structural"]
        assert not structural_flags

    def test_missing_warrant_is_invalid(self) -> None:
        graph = _make_graph(warrant_text="")
        result = validate(graph)
        assert result.status == ValidationStatus.INVALID
        assert any(f.type == "missing_required" and f.location == "warrant" for f in result.flags)

    def test_missing_backing_is_warning(self) -> None:
        graph = _make_graph(backing=None)
        result = validate(graph)
        assert any(f.type == "missing_backing" for f in result.flags)
        assert result.status != ValidationStatus.INVALID  # warning, not fatal

    def test_missing_qualifier_is_warning(self) -> None:
        graph = _make_graph(qualifier=None)
        result = validate(graph)
        assert any(f.type == "missing_qualifier" for f in result.flags)

    def test_missing_rebuttals_is_warning(self) -> None:
        graph = _make_graph(rebuttals=[])
        result = validate(graph)
        assert any(f.type == "missing_rebuttals" for f in result.flags)

    def test_implicit_warrant_flagged(self) -> None:
        graph = _make_graph()
        graph.warrant = Warrant(
            text="Something implicit",
            inference_type=InferenceType.INDUCTIVE,
            explicit=False,
        )
        result = validate(graph)
        assert any(f.type == "implicit_warrant" for f in result.flags)


# ---------------------------------------------------------------------------
# Pass 2: Inferential type validation — Deductive
# ---------------------------------------------------------------------------


class TestDeductiveValidation:
    """Paired tests for deductive reasoning: valid forms vs formal fallacies."""

    def test_modus_ponens_valid(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="A, A→B ⊢ B",
        )
        result = validate(graph)
        # No deductive flags
        deductive_flags = [f for f in result.flags if f.category.value == "deductive"]
        assert not deductive_flags

    def test_modus_tollens_valid(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="¬B, A→B ⊢ ¬A",
        )
        result = validate(graph)
        deductive_flags = [f for f in result.flags if f.category.value == "deductive"]
        assert not deductive_flags

    def test_affirming_consequent_invalid(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="B, A→B ⊢ A",
        )
        result = validate(graph)
        assert result.status == ValidationStatus.INVALID
        assert any(f.type == "affirming_consequent" for f in result.flags)

    def test_denying_antecedent_invalid(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="¬A, A→B ⊢ ¬B",
        )
        result = validate(graph)
        assert result.status == ValidationStatus.INVALID
        assert any(f.type == "denying_antecedent" for f in result.flags)

    def test_missing_formalization_warned(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization=None,
        )
        result = validate(graph)
        assert any(f.type == "missing_formalization" for f in result.flags)

    def test_unrecognized_form_warned(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="X ∧ Y → Z",
        )
        result = validate(graph)
        assert any(f.type == "unrecognized_deductive_form" for f in result.flags)


# ---------------------------------------------------------------------------
# Pass 2: Inferential type validation — Inductive
# ---------------------------------------------------------------------------


class TestInductiveValidation:
    """Paired tests: valid induction vs hasty generalization."""

    def test_valid_induction_large_sample(self) -> None:
        graph = _make_graph(
            claim_scope=ClaimScope.GENERAL,
            claim_confidence=Confidence.PROBABLE,
            data=[
                Datum(
                    text="Survey of n=2400 users showed 87% improvement",
                    source=DataSource.ENCYCLOPEDIA,
                    strength=DataStrength.STRONG,
                ),
            ],
            qualifier=Qualifier(text="typically", strength=Confidence.PROBABLE),
        )
        result = validate(graph)
        assert not any(f.type == "hasty_generalization" for f in result.flags)

    def test_hasty_generalization_small_sample(self) -> None:
        graph = _make_graph(
            claim_scope=ClaimScope.UNIVERSAL,
            claim_confidence=Confidence.CERTAIN,
            data=[
                Datum(
                    text="3 blog posts reported improvement",
                    source=DataSource.INFERRED,
                    strength=DataStrength.ANECDOTAL,
                ),
            ],
            qualifier=Qualifier(text="always", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        assert any(f.type == "hasty_generalization" for f in result.flags)

    def test_anecdotal_overclaim_flagged(self) -> None:
        graph = _make_graph(
            claim_text="Every team should use Kubernetes",
            claim_scope=ClaimScope.UNIVERSAL,
            data=[
                Datum(
                    text="A friend told me Kubernetes solved all their scaling problems",
                    source=DataSource.INFERRED,
                    strength=DataStrength.ANECDOTAL,
                ),
            ],
            qualifier=Qualifier(text="always", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        assert any(f.type == "anecdotal_overclaim" for f in result.flags)

    def test_false_cause_temporal(self) -> None:
        graph = _make_graph(
            claim_text="The new deploy caused the outage",
            warrant_text="After the deploy, the outage happened, so the deploy led to the failure",
            inference_type=InferenceType.INDUCTIVE,
        )
        result = validate(graph)
        assert any(f.type == "false_cause" for f in result.flags)

    def test_misclassified_evidence_flagged(self) -> None:
        graph = _make_graph(
            data=[
                Datum(
                    text="Someone said on a blog post that it works great",
                    source=DataSource.INFERRED,
                    strength=DataStrength.STRONG,  # mislabeled!
                ),
            ],
        )
        result = validate(graph)
        assert any(f.type == "misclassified_evidence" for f in result.flags)


# ---------------------------------------------------------------------------
# Pass 2: Inferential type validation — Abductive
# ---------------------------------------------------------------------------


class TestAbductiveValidation:
    def test_single_explanation_flagged(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.ABDUCTIVE,
            rebuttals=[],  # no alternatives considered
        )
        result = validate(graph)
        assert any(f.type == "single_explanation" for f in result.flags)

    def test_with_alternatives_not_flagged(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.ABDUCTIVE,
            rebuttals=[
                Rebuttal(
                    text="An alternative explanation is that the cache was warm",
                    severity=RebuttalSeverity.SIGNIFICANT,
                    addressed=True,
                    response="Cache state was controlled in the experiment",
                ),
            ],
        )
        result = validate(graph)
        assert not any(f.type == "single_explanation" for f in result.flags)


# ---------------------------------------------------------------------------
# Pass 2: Inferential type validation — Analogical
# ---------------------------------------------------------------------------


class TestAnalogicalValidation:
    def test_surface_analogy_flagged(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.ANALOGICAL,
            warrant_text="Microservices look like biological cells",
        )
        result = validate(graph)
        assert any(f.type == "surface_analogy" for f in result.flags)

    def test_structural_analogy_not_flagged(self) -> None:
        graph = _make_graph(
            inference_type=InferenceType.ANALOGICAL,
            warrant_text="Microservices function as independent units with the same mechanism of failure isolation",
        )
        result = validate(graph)
        assert not any(f.type == "surface_analogy" for f in result.flags)


# ---------------------------------------------------------------------------
# Pass 3: Cross-reference integrity
# ---------------------------------------------------------------------------


class TestCrossReferenceIntegrity:
    def test_unaddressed_contradiction_flagged(self) -> None:
        graph = _make_graph(rebuttals=[])
        result = validate(
            graph,
            known_contradictions=["Evidence shows the opposite effect in production"],
        )
        assert any(f.type == "incomplete_rebuttal" for f in result.flags)
        assert any(f.severity == FlagSeverity.CRITICAL for f in result.flags if f.type == "incomplete_rebuttal")

    def test_addressed_contradiction_not_flagged(self) -> None:
        graph = _make_graph(
            rebuttals=[
                Rebuttal(
                    text="Evidence shows the opposite effect in production",
                    severity=RebuttalSeverity.CRITICAL,
                    addressed=True,
                    response="That evidence was from an older version",
                ),
            ],
        )
        result = validate(
            graph,
            known_contradictions=["Evidence shows the opposite effect in production"],
        )
        assert not any(f.type == "incomplete_rebuttal" for f in result.flags)

    def test_unused_support_noted(self) -> None:
        graph = _make_graph()
        result = validate(
            graph,
            known_support=["Additional study confirms the finding"],
        )
        assert any(f.type == "unused_support" for f in result.flags)
        # Info severity, not critical
        assert all(
            f.severity == FlagSeverity.INFO
            for f in result.flags
            if f.type == "unused_support"
        )


# ---------------------------------------------------------------------------
# Pass 4: Qualifier calibration
# ---------------------------------------------------------------------------


class TestQualifierCalibration:
    def test_overclaim_strong_evidence_but_too_confident(self) -> None:
        """Strong evidence with inductive inference shouldn't be 'certain'."""
        graph = _make_graph(
            inference_type=InferenceType.INDUCTIVE,
            data=[
                Datum(text="Evidence", source=DataSource.ENCYCLOPEDIA, strength=DataStrength.STRONG),
            ],
            qualifier=Qualifier(text="certainly", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        assert any(f.type == "overclaim" for f in result.flags)
        assert result.qualifier_calibration is not None
        # Strong inductive → probable, not certain
        assert result.qualifier_calibration.recommended == Confidence.PROBABLE

    def test_calibrated_qualifier_passes(self) -> None:
        """Probable qualifier with strong inductive evidence is correctly calibrated."""
        graph = _make_graph(
            inference_type=InferenceType.INDUCTIVE,
            data=[
                Datum(text="Evidence", source=DataSource.ENCYCLOPEDIA, strength=DataStrength.STRONG),
            ],
            qualifier=Qualifier(text="probably", strength=Confidence.PROBABLE),
        )
        result = validate(graph)
        assert not any(f.type == "overclaim" for f in result.flags)
        if result.qualifier_calibration:
            assert result.qualifier_calibration.calibrated

    def test_severe_overclaim_is_critical(self) -> None:
        """Certain claim on anecdotal inductive evidence → critical flag."""
        graph = _make_graph(
            inference_type=InferenceType.INDUCTIVE,
            data=[
                Datum(text="Evidence", source=DataSource.INFERRED, strength=DataStrength.ANECDOTAL),
            ],
            qualifier=Qualifier(text="certainly", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        overclaim_flags = [f for f in result.flags if f.type == "overclaim"]
        assert overclaim_flags
        assert overclaim_flags[0].severity == FlagSeverity.CRITICAL

    def test_underclaim_noted_as_info(self) -> None:
        """Speculative qualifier with strong deductive evidence → info note."""
        graph = _make_graph(
            inference_type=InferenceType.DEDUCTIVE,
            formalization="A, A→B ⊢ B",
            data=[
                Datum(text="Evidence", source=DataSource.ENCYCLOPEDIA, strength=DataStrength.STRONG),
            ],
            qualifier=Qualifier(text="maybe", strength=Confidence.SPECULATIVE),
        )
        result = validate(graph)
        assert any(f.type == "underclaim" for f in result.flags)
        underclaim = [f for f in result.flags if f.type == "underclaim"]
        assert underclaim[0].severity == FlagSeverity.INFO

    def test_analogical_evidence_caps_at_plausible(self) -> None:
        """Even strong evidence with analogical inference shouldn't exceed plausible."""
        graph = _make_graph(
            inference_type=InferenceType.ANALOGICAL,
            warrant_text="Functions as same mechanism",
            data=[
                Datum(text="Evidence", source=DataSource.ENCYCLOPEDIA, strength=DataStrength.STRONG),
            ],
            qualifier=Qualifier(text="certainly", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        assert result.qualifier_calibration is not None
        assert result.qualifier_calibration.recommended == Confidence.PLAUSIBLE


# ---------------------------------------------------------------------------
# Integration: compound cases
# ---------------------------------------------------------------------------


class TestCompoundCases:
    """Cases that trigger multiple validation passes simultaneously."""

    def test_the_canonical_pair_valid(self) -> None:
        """The spec's canonical example: valid PostgreSQL induction."""
        graph = ArgumentGraph(
            claim=Claim(
                text="Most PostgreSQL users will likely see performance improvements from upgrading",
                scope=ClaimScope.GENERAL,
                confidence=Confidence.PROBABLE,
            ),
            data=[
                Datum(
                    text="87% of surveyed PostgreSQL users (n=2400) reported improved query performance after upgrading to v16",
                    source=DataSource.ENCYCLOPEDIA,
                    strength=DataStrength.STRONG,
                ),
            ],
            warrant=Warrant(
                text="Large representative sample generalizes to the broader population",
                inference_type=InferenceType.INDUCTIVE,
            ),
            backing=Backing(
                text="Standard statistical generalization from representative samples",
                source=BackingSource.EMPIRICAL_RESEARCH,
            ),
            qualifier=Qualifier(text="typically", strength=Confidence.PROBABLE),
            rebuttals=[
                Rebuttal(
                    text="Some workloads are atypical",
                    severity=RebuttalSeverity.MINOR,
                    addressed=True,
                    response="The qualifier 'most' accounts for this",
                ),
            ],
        )
        result = validate(graph)
        assert result.status == ValidationStatus.VALID

    def test_the_canonical_pair_invalid(self) -> None:
        """The spec's canonical example: hasty PostgreSQL generalization."""
        graph = ArgumentGraph(
            claim=Claim(
                text="Upgrading to PostgreSQL 16 always improves performance",
                scope=ClaimScope.UNIVERSAL,
                confidence=Confidence.CERTAIN,
            ),
            data=[
                Datum(
                    text="3 blog posts reported improved performance after upgrading",
                    source=DataSource.INFERRED,
                    strength=DataStrength.ANECDOTAL,
                ),
            ],
            warrant=Warrant(
                text="Blog reports indicate universal improvement",
                inference_type=InferenceType.INDUCTIVE,
            ),
            qualifier=Qualifier(text="always", strength=Confidence.CERTAIN),
        )
        result = validate(graph)
        assert result.status == ValidationStatus.FLAGGED
        assert any(f.type == "hasty_generalization" for f in result.flags)
        assert any(f.type == "overclaim" for f in result.flags)

    def test_everything_wrong(self) -> None:
        """A maximally broken argument should produce many flags."""
        graph = ArgumentGraph(
            claim=Claim(
                text="Every developer should always use Rust",
                scope=ClaimScope.UNIVERSAL,
                confidence=Confidence.CERTAIN,
            ),
            data=[
                Datum(
                    text="A friend told me on a blog post that Rust is fast",
                    source=DataSource.INFERRED,
                    strength=DataStrength.STRONG,  # mislabeled
                ),
            ],
            warrant=Warrant(
                text="Fast languages are the best for everything",
                inference_type=InferenceType.INDUCTIVE,
                explicit=False,
            ),
        )
        # No backing, no qualifier, no rebuttals
        result = validate(graph)
        assert result.status == ValidationStatus.FLAGGED
        assert len(result.flags) >= 4  # multiple issues

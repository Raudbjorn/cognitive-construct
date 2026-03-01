"""Toulmin argumentation engine — symbolic validation for informal discourse.

Public API:
    validate()          — Four-pass validation of an ArgumentGraph
    decompose()         — LLM-powered intent → Toulmin graph decomposition
    select_strategy()   — Audience-based delivery strategy selection
    generate_analogy()  — Structural analogy via diffusion LLM
    RhetoricEngine      — Full pipeline orchestrator

Models:
    ArgumentGraph, Claim, Datum, Warrant, Backing, Qualifier, Rebuttal
    ValidationResult, AudienceModel, Analogy
    RhetoricPlan, EngineError, DecompositionError, BridgeError
"""

from toulmin.models import (
    Analogy,
    ArgumentGraph,
    AudienceModel,
    Backing,
    Claim,
    Datum,
    Qualifier,
    Rebuttal,
    ValidationResult,
    Warrant,
)
from toulmin.validate import validate

__all__ = [
    # Core validation
    "validate",
    # Models
    "ArgumentGraph",
    "AudienceModel",
    "Analogy",
    "Backing",
    "Claim",
    "Datum",
    "Qualifier",
    "Rebuttal",
    "ValidationResult",
    "Warrant",
]

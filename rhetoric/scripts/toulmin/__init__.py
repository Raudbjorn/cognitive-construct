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

from toulmin.bridge import BridgeError, generate_analogy
from toulmin.decompose import DecompositionError, decompose
from toulmin.engine import EngineError, RhetoricEngine, RhetoricPlan
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
from toulmin.strategy import select_strategy
from toulmin.validate import validate

__all__ = [
    # Core functions
    "validate",
    "decompose",
    "select_strategy",
    "generate_analogy",
    # Engine
    "RhetoricEngine",
    "RhetoricPlan",
    # Error types
    "EngineError",
    "DecompositionError",
    "BridgeError",
    # Models
    "Analogy",
    "ArgumentGraph",
    "AudienceModel",
    "Backing",
    "Claim",
    "Datum",
    "Qualifier",
    "Rebuttal",
    "ValidationResult",
    "Warrant",
]

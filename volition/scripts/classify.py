"""Intent Classification Engine for Volition.

Replaces keyword-counting _classify_intent() with a two-signal fusion system:
1. Embedding similarity (Model2Vec via shared.embeddings)
2. Keyword scoring (preserved from original)

Fused via shared.fusion.RRFEngine with optional feedback adjustment from
shared.feedback.FeedbackCollector.

See SPEC.md Section 3.3 for design rationale.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure shared is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from shared.embeddings import ScoredItem, rank_by_relevance
from shared.fusion import RRFEngine, RRFConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentPrototype:
    """Handler registration: describes what a handler does.

    Attributes:
        handler: Handler identifier (e.g. "code_edit", "llm_call").
        prototype_text: Natural-language description embedded at startup.
        keywords: Secondary signal keyword list.
        risk_level: LOW, MEDIUM, or HIGH.
        fallback_chain: Ordered list of fallback handler ids.
        requires_confirmation: Whether the handler always needs --confirm.
    """

    handler: str
    prototype_text: str
    keywords: tuple[str, ...] = ()
    risk_level: str = "LOW"
    fallback_chain: tuple[str, ...] = ()
    requires_confirmation: bool = False


@dataclass
class CandidateScore:
    """Per-handler classification scores."""

    handler: str
    embedding_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    feedback_adjustment: float = 0.0


@dataclass
class ClassificationResult:
    """Full classification output with all candidates."""

    action: str
    candidates: list[CandidateScore] = field(default_factory=list)
    selected: str = "llm_call"
    confidence: float = 0.0
    above_threshold: bool = False
    feedback_adjustment: float = 0.0


# ---------------------------------------------------------------------------
# Prototype registry
# ---------------------------------------------------------------------------

DEFAULT_PROTOTYPES: tuple[IntentPrototype, ...] = (
    IntentPrototype(
        handler="code_edit",
        prototype_text=(
            "Edit, refactor, modify, or fix source code symbols and files "
            "using LSP-powered semantic editing. Improve code quality, "
            "rename symbols, extract methods, change implementations."
        ),
        keywords=(
            "refactor", "edit", "modify", "add", "remove", "fix",
            "update", "change", "rename", "extract", "implement",
            "rewrite", "improve", "secure",
        ),
        risk_level="MEDIUM",
        fallback_chain=("text_edit",),
    ),
    IntentPrototype(
        handler="llm_call",
        prototype_text=(
            "Consult a large language model to explain, analyze, review, "
            "suggest improvements, answer questions, or provide advice "
            "about code, architecture, or development practices."
        ),
        keywords=(
            "explain", "analyze", "review", "suggest", "consult",
            "help", "advise", "summarize", "describe", "why",
            "how", "what",
        ),
        risk_level="LOW",
    ),
    IntentPrototype(
        handler="web_search",
        prototype_text=(
            "Search the web for documentation, tutorials, current "
            "information, news, API references, or solutions to "
            "programming problems and technical questions."
        ),
        keywords=(
            "search", "find", "lookup", "what is", "latest",
            "current", "news", "documentation", "docs", "tutorial",
        ),
        risk_level="LOW",
    ),
    IntentPrototype(
        handler="security",
        prototype_text=(
            "Perform security reconnaissance, vulnerability scanning, "
            "and internet-facing service enumeration using Shodan. "
            "Identify exposed services, open ports, and security risks."
        ),
        keywords=(
            "scan", "expose", "vulnerability", "shodan", "security",
            "recon", "port", "enumerate", "attack surface",
        ),
        risk_level="HIGH",
        requires_confirmation=True,
    ),
)


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------

# Configurable via environment
_CONFIDENCE_THRESHOLD = float(os.environ.get("VOLITION_CONFIDENCE_THRESHOLD", "0.65"))
_EMBEDDING_WEIGHT = float(os.environ.get("VOLITION_EMBEDDING_WEIGHT", "0.7"))
_KEYWORD_WEIGHT = float(os.environ.get("VOLITION_KEYWORD_WEIGHT", "0.3"))
_LEARNING_RATE = float(os.environ.get("VOLITION_LEARNING_RATE", "0.2"))


def _keyword_score(action: str, prototype: IntentPrototype) -> float:
    """Compute fractional keyword overlap score."""
    if not prototype.keywords:
        return 0.0
    action_lower = action.lower()
    hits = sum(1 for kw in prototype.keywords if kw in action_lower)
    return hits / len(prototype.keywords)


def _get_feedback_scores() -> dict[str, float]:
    """Get handler feedback scores, returning empty dict on failure."""
    try:
        from shared.feedback import FeedbackCollector

        return FeedbackCollector.get_instance().get_source_scores()
    except ImportError:
        return {}
    except Exception:
        logger.debug("Feedback score retrieval failed", exc_info=True)
        return {}


def classify_intent(
    action: str,
    prototypes: tuple[IntentPrototype, ...] | None = None,
    threshold: float | None = None,
    use_feedback: bool = True,
) -> ClassificationResult:
    """Classify user action intent via embedding + keyword fusion.

    Args:
        action: Natural-language action description.
        prototypes: Handler prototypes to classify against.
            Defaults to DEFAULT_PROTOTYPES.
        threshold: Confidence threshold. Defaults to env var or 0.65.
        use_feedback: Whether to apply feedback adjustment.

    Returns:
        ClassificationResult with all candidates and scores.
    """
    if prototypes is None:
        prototypes = DEFAULT_PROTOTYPES
    if threshold is None:
        threshold = _CONFIDENCE_THRESHOLD

    if not action.strip():
        return ClassificationResult(action=action)

    # --- Signal 1: Embedding similarity ---
    embedding_scores: dict[str, float] = {}
    scored_items: list[ScoredItem] = rank_by_relevance(
        items=list(prototypes),
        query=action,
        key=lambda p: p.prototype_text,
    )
    for si in scored_items:
        proto: IntentPrototype = si.item
        embedding_scores[proto.handler] = max(si.score, 0.0)

    # --- Signal 2: Keyword scoring ---
    keyword_scores: dict[str, float] = {}
    for proto in prototypes:
        keyword_scores[proto.handler] = _keyword_score(action, proto)

    # --- Fusion via RRF ---
    # Build ordered lists for each signal (best first)
    embedding_ranked = sorted(
        prototypes, key=lambda p: embedding_scores.get(p.handler, 0.0), reverse=True,
    )
    keyword_ranked = sorted(
        prototypes, key=lambda p: keyword_scores.get(p.handler, 0.0), reverse=True,
    )

    engine = RRFEngine(RRFConfig(k=60, normalize_scores=True, max_results=len(prototypes)))
    fused_results = engine.fuse_sources(
        source_results={
            "embedding": list(embedding_ranked),
            "keyword": list(keyword_ranked),
        },
        source_weights={
            "embedding": _EMBEDDING_WEIGHT,
            "keyword": _KEYWORD_WEIGHT,
        },
        key_fn=lambda p: p.handler,
    )

    # Build candidate list
    fused_map: dict[str, float] = {}
    for fr in fused_results:
        fused_map[fr.item.handler] = fr.score

    # --- Feedback adjustment (Constitution Rule 5: never override safety) ---
    feedback_scores = _get_feedback_scores() if use_feedback else {}

    candidates: list[CandidateScore] = []
    for proto in prototypes:
        raw_fused = fused_map.get(proto.handler, 0.0)

        # Feedback: mild +-10% adjustment (spec section 3.3)
        handler_feedback = feedback_scores.get(proto.handler, 0.5)
        adjustment = (handler_feedback - 0.5) * _LEARNING_RATE
        adjusted_score = raw_fused * (1.0 + adjustment)

        # Constitution Rule 5: raw_fused_score must independently exceed threshold
        # Feedback cannot push a below-threshold score above threshold
        candidates.append(CandidateScore(
            handler=proto.handler,
            embedding_score=embedding_scores.get(proto.handler, 0.0),
            keyword_score=keyword_scores.get(proto.handler, 0.0),
            fused_score=adjusted_score,
            feedback_adjustment=adjustment,
        ))

    # Sort by fused score descending
    candidates.sort(key=lambda c: c.fused_score, reverse=True)

    # Select winner
    top = candidates[0] if candidates else None
    if top is None:
        return ClassificationResult(action=action)

    # Constitution Rule 5: check raw fused score for threshold
    raw_top_fused = fused_map.get(top.handler, 0.0)
    above = raw_top_fused >= threshold

    return ClassificationResult(
        action=action,
        candidates=candidates,
        selected=top.handler if above else "clarification",
        confidence=top.fused_score,
        above_threshold=above,
        feedback_adjustment=top.feedback_adjustment,
    )


def format_clarification(result: ClassificationResult) -> dict[str, Any]:
    """Build a clarification request from an ambiguous classification.

    Returns a dict suitable for JSON output.
    """
    top_two = result.candidates[:2]
    return {
        "status": "clarification_required",
        "message": "Intent is ambiguous. Did you mean one of these?",
        "candidates": [
            {
                "handler": c.handler,
                "confidence": round(c.fused_score, 3),
                "embedding_score": round(c.embedding_score, 3),
                "keyword_score": round(c.keyword_score, 3),
            }
            for c in top_two
        ],
        "action": result.action,
    }

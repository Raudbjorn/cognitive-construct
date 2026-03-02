"""Query preprocessing pipeline.

Processes raw user queries through normalization, typo correction,
synonym expansion, and classification before they reach search backends.

Ported from TTTTRPS's ``preprocess/pipeline.rs`` pattern. Key insight
preserved: expand synonyms for keyword search, but use the corrected-only
form for semantic/embedding search (synonym noise degrades vector recall).

Usage:
    from shared.query_pipeline import QueryPipeline

    pipeline = QueryPipeline.default()
    result = pipeline.process("kuberntes deploymnet")
    # result.corrected == "kubernetes deployment"
    # result.text_for_embedding == "kubernetes deployment"
    # result.expanded.expanded_text == "kubernetes k8s deployment deploy ..."
    # result.query_type == "general_search"
    # result.corrections == [Correction("kuberntes", "kubernetes", 2), ...]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .vocabulary import Correction, ExpandedQuery, Vocabulary

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

_REPO_HINT_RE = re.compile(r"repo:([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)")
_URL_RE = re.compile(r"https?://")
_CODE_PATTERNS = [
    re.compile(r"\bdef\s+\w+"),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bfunction\s+\w+"),
    re.compile(r"\bimport\s+"),
    re.compile(r"\bfrom\s+\w+\s+import\b"),
]
_TIME_KEYWORDS = frozenset([
    "latest", "current", "2024", "2025", "2026",
    "recent", "news", "update", "today", "now",
])
_LIBRARY_KEYWORDS = frozenset([
    "how to use", "api", "documentation", "docs", "example", "tutorial",
    "guide", "reference", "getting started", "quickstart",
])

# Explicit prefix overrides (highest priority routing)
_TYPE_PREFIXES: dict[str, str] = {
    "doc:": "library_docs",
    "docs:": "library_docs",
    "code:": "code_context",
    "web:": "general_search",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the query preprocessing pipeline."""

    enable_typo_correction: bool = True
    enable_synonym_expansion: bool = True
    enable_classification: bool = True
    min_typo_word_length: int = 3
    """Don't attempt typo correction on words shorter than this."""


# ------------------------------------------------------------------
# Pipeline result
# ------------------------------------------------------------------

@dataclass
class ProcessedQuery:
    """Result of running a query through the preprocessing pipeline.

    Consumers should use:
    - ``text_for_embedding`` for semantic/AI sources (Perplexity, Context7)
    - ``expanded`` for keyword-oriented sources (Exa, Kagi, SearXNG)
    - ``query_type`` for routing decisions
    - ``corrections`` / ``suggestions`` for UI feedback
    """

    original: str
    """Raw user input."""

    corrected: str
    """Normalized and typo-corrected query."""

    corrections: list[Correction]
    """Individual typo corrections applied (for "did you mean" UI)."""

    expanded: ExpandedQuery
    """Synonym-expanded query for keyword search backends."""

    text_for_embedding: str
    """Corrected (NOT expanded) text for vector/semantic search.
    Synonym expansion adds noise to embeddings."""

    query_type: str
    """Classified query type for routing: library_docs | general_search | code_context | repository."""

    repo_hint: str | None
    """Extracted ``repo:owner/name`` hint, if present."""

    cleaned_query: str
    """Query with repo hint and type prefix stripped."""

    suggestions: list[str] = field(default_factory=list)
    """Search tips for the user (e.g. "Try also: ...")."""

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0

    @property
    def corrections_summary(self) -> str | None:
        """Human-readable correction summary, or None if no corrections."""
        if not self.corrections:
            return None
        parts = [f"{c.original} → {c.corrected}" for c in self.corrections]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output (e.g. in Encyclopedia response)."""
        result: dict[str, Any] = {
            "original": self.original,
            "corrected": self.corrected,
            "text_for_embedding": self.text_for_embedding,
            "expanded_text": self.expanded.expanded_text,
            "was_expanded": self.expanded.was_expanded,
            "query_type": self.query_type,
        }
        if self.corrections:
            result["corrections"] = [
                {"from": c.original, "to": c.corrected, "distance": c.edit_distance}
                for c in self.corrections
            ]
        if self.suggestions:
            result["suggestions"] = self.suggestions
        if self.repo_hint:
            result["repo_hint"] = self.repo_hint
        return result


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class QueryPipeline:
    """Multi-stage query preprocessing pipeline.

    Stages (in order):
    1. Normalize: trim, collapse whitespace, lowercase
    2. Extract metadata: repo hints, type prefixes
    3. Typo correction: static table + edit-distance fallback
    4. Synonym expansion: multi-way + one-way, respects max_expansions
    5. Classify: route to appropriate search backends
    6. Generate tips: suggest related searches

    The pipeline never raises. Failures in any stage are logged and
    that stage is skipped (graceful degradation).
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        config: PipelineConfig | None = None,
    ) -> None:
        self._vocab = vocabulary
        self._config = config or PipelineConfig()

    @classmethod
    def default(cls) -> QueryPipeline:
        """Create a pipeline with the default dev/programming vocabulary."""
        return cls(vocabulary=Vocabulary.load_default())

    @classmethod
    def with_vocabulary_file(cls, path: str) -> QueryPipeline:
        """Create a pipeline with a custom vocabulary JSON file."""
        from pathlib import Path as P
        return cls(vocabulary=Vocabulary.load_from_json(P(path)))

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocab

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, raw_query: str) -> ProcessedQuery:
        """Process a raw user query through all pipeline stages.

        This is the primary API. Returns a ``ProcessedQuery`` with all
        the information downstream consumers need.
        """
        # 1. Normalize
        normalized = _normalize(raw_query)
        if not normalized:
            return _empty_result(raw_query)

        # 2. Extract repo hint and type prefix
        repo_hint, after_repo = _extract_repo_hint(normalized)
        query_type_override, cleaned = _strip_type_prefix(after_repo)

        # 3. Typo correction
        corrected, corrections = self._correct(cleaned)

        # 4. Synonym expansion (on corrected text)
        if self._config.enable_synonym_expansion:
            expanded = self._vocab.expand_query(corrected)
        else:
            expanded = ExpandedQuery(
                original=corrected,
                term_groups=[[w] for w in corrected.split()],
            )

        # 5. Classify
        if query_type_override:
            query_type = query_type_override
        elif repo_hint:
            query_type = "code_context"
        elif self._config.enable_classification:
            query_type = _classify(cleaned)
        else:
            query_type = "general_search"

        # 6. Generate suggestions
        suggestions = self._generate_suggestions(corrected, expanded)

        return ProcessedQuery(
            original=raw_query,
            corrected=corrected,
            corrections=corrections,
            expanded=expanded,
            text_for_embedding=corrected,
            query_type=query_type,
            repo_hint=repo_hint,
            cleaned_query=cleaned,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _correct(self, query: str) -> tuple[str, list[Correction]]:
        """Apply typo correction to each word in the query."""
        if not self._config.enable_typo_correction:
            return query, []

        words = query.split()
        corrected_words: list[str] = []
        corrections: list[Correction] = []

        for word in words:
            if len(word) < self._config.min_typo_word_length:
                corrected_words.append(word)
                continue

            correction = self._vocab.correct_typo(word)
            if correction is not None:
                corrected_words.append(correction)
                corrections.append(Correction(
                    original=word,
                    corrected=correction,
                    edit_distance=_quick_distance(word, correction),
                ))
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), corrections

    def _generate_suggestions(
        self,
        corrected: str,
        expanded: ExpandedQuery,
    ) -> list[str]:
        """Generate search tips based on expansion results."""
        suggestions: list[str] = []

        # If we expanded abbreviations, tell the user
        for group in expanded.term_groups:
            if len(group) > 1:
                original = group[0]
                # Only suggest if the original looks like an abbreviation (short, all lower/upper)
                if len(original) <= 5 and original.isalpha():
                    alternatives = [g for g in group[1:] if g != original][:2]
                    if alternatives:
                        suggestions.append(
                            f"{original.upper()} = {', '.join(alternatives)}"
                        )

        return suggestions[:3]  # Cap at 3 suggestions


# ------------------------------------------------------------------
# Pure functions (no self, easily testable)
# ------------------------------------------------------------------

def _normalize(query: str) -> str:
    """Trim, collapse whitespace, lowercase."""
    return " ".join(query.split()).lower()


def _extract_repo_hint(query: str) -> tuple[str | None, str]:
    """Extract ``repo:owner/name`` and return (hint, cleaned_query)."""
    match = _REPO_HINT_RE.search(query)
    if not match:
        return None, query
    repo = match.group(1)
    cleaned = (query[: match.start()] + query[match.end() :]).strip()
    # Collapse any double spaces left by the removal
    cleaned = " ".join(cleaned.split())
    return repo, cleaned or query


def _strip_type_prefix(query: str) -> tuple[str | None, str]:
    """Strip explicit type prefix (doc:, code:, web:) and return (type, cleaned)."""
    for prefix, qtype in _TYPE_PREFIXES.items():
        if query.startswith(prefix):
            return qtype, query[len(prefix) :].strip()
    return None, query


def _classify(query: str) -> str:
    """Classify a query into a routing type.

    Priority order:
    1. URL pattern → general_search
    2. Code patterns → code_context
    3. Time-sensitive keywords → general_search
    4. Library/doc keywords → library_docs
    5. Default → library_docs
    """
    if _URL_RE.search(query):
        return "general_search"

    if any(p.search(query) for p in _CODE_PATTERNS):
        return "code_context"

    words = set(query.split())
    if words & _TIME_KEYWORDS:
        return "general_search"

    if any(kw in query for kw in _LIBRARY_KEYWORDS):
        return "library_docs"

    return "library_docs"


def _quick_distance(a: str, b: str) -> int:
    """Quick edit distance for correction metadata.

    Uses the same Levenshtein from vocabulary module but imported
    to avoid exposing private functions.
    """
    from .vocabulary import _levenshtein
    return _levenshtein(a.lower(), b.lower())


def _empty_result(raw: str) -> ProcessedQuery:
    """Return a no-op result for empty queries."""
    return ProcessedQuery(
        original=raw,
        corrected="",
        corrections=[],
        expanded=ExpandedQuery(original="", term_groups=[]),
        text_for_embedding="",
        query_type="general_search",
        repo_hint=None,
        cleaned_query="",
    )

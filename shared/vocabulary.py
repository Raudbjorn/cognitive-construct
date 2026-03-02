"""Domain vocabulary for query preprocessing.

Provides synonym expansion, abbreviation resolution, typo correction,
and stop-word filtering. Vocabulary is loaded from JSON files, making
it easy to swap domains (dev/programming, film/location, TTRPG, etc.).

Ported from the Rust QueryExpander/SynonymMap pattern in TTTTRPS.

Usage:
    from shared.vocabulary import Vocabulary

    vocab = Vocabulary.load_default()
    expanded = vocab.expand_term("api")
    # ["api", "application programming interface"]

    corrected = vocab.correct_typo("kuberntes")
    # "kubernetes"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VOCAB_DIR = Path(__file__).parent / "vocabularies"
_DEFAULT_VOCAB = "dev_programming.json"


@dataclass(frozen=True)
class Correction:
    """A single typo correction applied to a query term."""

    original: str
    corrected: str
    edit_distance: int


@dataclass
class ExpandedQuery:
    """Result of synonym-expanding a full query.

    Each element in ``term_groups`` is a list where the first item is the
    original (possibly corrected) term and remaining items are synonyms.

    For keyword search: OR within groups, AND between groups.
    For embedding search: use only the first term from each group
    (``text_for_embedding``).
    """

    original: str
    term_groups: list[list[str]]

    @property
    def text_for_embedding(self) -> str:
        """Corrected query without synonym noise — use for vector search."""
        return " ".join(group[0] for group in self.term_groups if group)

    @property
    def expanded_text(self) -> str:
        """Flat expanded query with all synonyms — use for keyword search."""
        parts: list[str] = []
        for group in self.term_groups:
            parts.extend(group)
        return " ".join(parts)

    @property
    def was_expanded(self) -> bool:
        return any(len(group) > 1 for group in self.term_groups)


class Vocabulary:
    """Domain-specific vocabulary for query preprocessing.

    Supports:
    - Multi-way synonyms (all terms interchangeable)
    - One-way synonyms (source expands to targets, not reverse)
    - Abbreviation resolution (subset of multi-way)
    - Known-term recognition
    - Typo correction via edit distance
    - Stop-word filtering
    """

    def __init__(
        self,
        multi_way: list[set[str]] | None = None,
        one_way: dict[str, list[str]] | None = None,
        known_terms: set[str] | None = None,
        stop_words: set[str] | None = None,
        common_misspellings: dict[str, str] | None = None,
        max_expansions: int = 6,
    ) -> None:
        self._multi_way: list[set[str]] = multi_way or []
        self._one_way: dict[str, list[str]] = one_way or {}
        self._known_terms: set[str] = known_terms or set()
        self._stop_words: set[str] = stop_words or set()
        self._misspellings: dict[str, str] = common_misspellings or {}
        self._max_expansions = max_expansions

        # Build a fast lookup: term → index into _multi_way
        self._term_to_group: dict[str, int] = {}
        for idx, group in enumerate(self._multi_way):
            for term in group:
                self._term_to_group[term] = idx

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_from_json(cls, path: Path) -> Vocabulary:
        """Load vocabulary from a JSON file.

        Returns an empty vocabulary on any IO/parse error (never raises).
        """
        try:
            with open(path) as f:
                data: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to load vocabulary from %s: %s",
                path,
                exc,
            )
            return cls()

        return cls._from_dict(data)

    @classmethod
    def load_default(cls) -> Vocabulary:
        """Load the default dev/programming vocabulary."""
        return cls.load_from_json(_VOCAB_DIR / _DEFAULT_VOCAB)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        max_exp = data.get("max_expansions", 6)

        # Parse multi-way synonym groups
        multi_way: list[set[str]] = []
        for key, values in data.get("multi_way", {}).items():
            group: set[str] = {key.lower()}
            for v in values:
                group.add(v.lower())
            multi_way.append(group)

        # Parse one-way synonyms
        one_way: dict[str, list[str]] = {}
        for source, targets in data.get("one_way", {}).items():
            one_way[source.lower()] = [t.lower() for t in targets]

        # Parse known terms
        known_terms: set[str] = set()
        for term in data.get("known_terms", []):
            known_terms.add(term.lower())
        # Also add all multi-way and one-way terms as known
        for group in multi_way:
            known_terms.update(group)
        for source, targets in one_way.items():
            known_terms.add(source)
            known_terms.update(targets)

        stop_words = {w.lower() for w in data.get("stop_words", [])}
        misspellings = {
            k.lower(): v.lower()
            for k, v in data.get("common_misspellings", {}).items()
        }

        return cls(
            multi_way=multi_way,
            one_way=one_way,
            known_terms=known_terms,
            stop_words=stop_words,
            common_misspellings=misspellings,
            max_expansions=max_exp,
        )

    # ------------------------------------------------------------------
    # Runtime additions
    # ------------------------------------------------------------------

    def add_multi_way(self, terms: list[str]) -> None:
        """Add a multi-way synonym group at runtime."""
        group = {t.lower() for t in terms}
        if len(group) < 2:
            return

        # Find all existing groups that overlap with the new terms
        overlapping_idxs = [
            idx for idx, existing in enumerate(self._multi_way)
            if group & existing
        ]

        if overlapping_idxs:
            # Merge into the first overlapping group
            target_idx = overlapping_idxs[0]
            merged = self._multi_way[target_idx] | group
            for idx in overlapping_idxs[1:]:
                merged |= self._multi_way[idx]
                self._multi_way[idx] = set()  # empty, preserve indices
            self._multi_way[target_idx] = merged
            for t in merged:
                self._term_to_group[t] = target_idx
            self._known_terms.update(merged)
        else:
            new_idx = len(self._multi_way)
            self._multi_way.append(group)
            for t in group:
                self._term_to_group[t] = new_idx
            self._known_terms.update(group)

    def add_one_way(self, source: str, targets: list[str]) -> None:
        """Add a one-way synonym mapping at runtime."""
        src = source.lower()
        tgts = [t.lower() for t in targets]
        self._one_way.setdefault(src, []).extend(tgts)
        self._known_terms.add(src)
        self._known_terms.update(tgts)

    # ------------------------------------------------------------------
    # Term operations
    # ------------------------------------------------------------------

    def expand_term(self, term: str) -> list[str]:
        """Expand a single term to itself + synonyms.

        Returns a list where the first element is always the original term.
        Limited by ``max_expansions``.
        """
        lower = term.lower()
        result: list[str] = [lower]

        # Multi-way lookup
        group_idx = self._term_to_group.get(lower)
        if group_idx is not None:
            for synonym in self._multi_way[group_idx]:
                if synonym != lower and synonym not in result:
                    result.append(synonym)
                    if len(result) >= self._max_expansions:
                        return result

        # One-way lookup
        targets = self._one_way.get(lower)
        if targets:
            for target in targets:
                if target not in result:
                    result.append(target)
                    if len(result) >= self._max_expansions:
                        return result

        return result

    def expand_query(self, query: str) -> ExpandedQuery:
        """Expand all terms in a query.

        Each word becomes a group of [original, synonym1, synonym2, ...].
        """
        words = query.split()
        groups = [self.expand_term(w) for w in words]
        return ExpandedQuery(original=query, term_groups=groups)

    def correct_typo(self, term: str) -> str | None:
        """Return corrected spelling, or None if no correction found.

        Checks the static misspellings table first (O(1)), then falls
        back to edit-distance search against known terms.
        """
        lower = term.lower()

        # Already a known term — no correction needed
        if lower in self._known_terms:
            return None

        # Common inflections of known terms are valid (plurals, -ed, -ing)
        if _is_inflection_of_known(lower, self._known_terms):
            return None

        # Check static misspellings table
        static_correction = self._misspellings.get(lower)
        if static_correction:
            return static_correction

        # Edit-distance fallback against known terms
        max_dist = 1 if len(lower) <= 4 else 2
        best_match: str | None = None
        best_dist = max_dist + 1

        for known in self._known_terms:
            # Quick length filter
            if abs(len(known) - len(lower)) > max_dist:
                continue
            dist = _levenshtein(lower, known)
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_match = known

        return best_match

    def is_known_term(self, term: str) -> bool:
        return term.lower() in self._known_terms

    def is_stop_word(self, term: str) -> bool:
        return term.lower() in self._stop_words

    def filter_stop_words(self, terms: list[str]) -> list[str]:
        """Remove stop words from a list of terms."""
        return [t for t in terms if not self.is_stop_word(t)]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "multi_way_groups": len(self._multi_way),
            "one_way_sources": len(self._one_way),
            "known_terms": len(self._known_terms),
            "stop_words": len(self._stop_words),
            "misspellings": len(self._misspellings),
        }


# ------------------------------------------------------------------
# Inflection recognition
# ------------------------------------------------------------------

_INFLECTION_SUFFIXES = ("s", "es", "ed", "ing", "er", "est", "ly")


def _is_inflection_of_known(word: str, known: set[str]) -> bool:
    """Check if word is a common English inflection of a known term.

    Prevents the typo corrector from "fixing" plurals and verb forms.
    E.g., "hooks" should not be corrected to "hook" when "hook" is known.
    """
    for suffix in _INFLECTION_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stem = word[: -len(suffix)]
            if stem in known:
                return True
            # Handle consonant doubling: "running" → "run"
            if len(stem) > 1 and stem[-1] == stem[-2]:
                if stem[:-1] in known:
                    return True
    # Handle -ies → -y: "queries" → "query"
    if word.endswith("ies") and len(word) > 4:
        if (word[:-3] + "y") in known:
            return True
    return False


# ------------------------------------------------------------------
# Edit distance (pure Python, no external dependency)
# ------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Optimize: use single-row DP
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)

    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[len(b)]

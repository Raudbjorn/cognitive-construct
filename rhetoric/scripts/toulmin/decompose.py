"""Constrained decomposition of natural language intent into Toulmin graphs.

This module solves the bootstrap problem: the agent needs to decompose its own
reasoning into a formal structure, but can't reliably do so via open-ended
generation. The solution: a rigid template with forced choices (enums) that
the validation engine can then check.

The LLM fills slots. The schema enforces structure. The validator checks logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

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
    InferenceType,
    Qualifier,
    Rebuttal,
    RebuttalSeverity,
    Warrant,
)

logger = logging.getLogger(__name__)

# The constrained template prompt. Enum values are listed explicitly so the
# model can only choose from valid options. "MISSING" is a valid response
# for optional fields — it's detectable, unlike silent omission.
_DECOMPOSITION_SYSTEM = """\
You are a logical analysis engine. Your ONLY job is to decompose an argument \
into its Toulmin structure. Respond with ONLY valid JSON matching the schema below. \
No preamble, no markdown, no explanation.

If you cannot determine a field, use "MISSING" as the string value. \
Do NOT fabricate content — "MISSING" is always preferable to hallucination.

JSON Schema:
{
  "claim_text": "string — the specific conclusion being argued",
  "claim_scope": "universal | general | particular | existential",
  "claim_confidence": "certain | probable | plausible | possible | speculative",

  "data": [
    {
      "text": "string — a piece of evidence",
      "source": "encyclopedia | user_stated | common_knowledge | inferred",
      "strength": "strong | moderate | weak | anecdotal"
    }
  ],

  "warrant_text": "string — WHY the data supports the claim (the bridge)",
  "warrant_type": "deductive | inductive | abductive | analogical",
  "warrant_formalization": "string | null — logical form if deductive, e.g. 'A, A→B ⊢ B'",

  "backing_text": "string — what supports the warrant itself",
  "backing_source": "encyclopedia | domain_expertise | empirical_research | authority",

  "qualifier_text": "string — hedging language for the claim",
  "qualifier_strength": "certain | probable | plausible | possible | speculative",

  "rebuttals": [
    {
      "text": "string — a counterargument",
      "severity": "critical | significant | minor",
      "response": "string | null — how the argument addresses this"
    }
  ]
}"""


@dataclass(frozen=True)
class DecompositionError:
    """Represents a failure in the decomposition process."""

    message: str
    raw_response: str | None = None


async def decompose(
    intent: str,
    *,
    api_base: str = "https://api.inceptionlabs.ai",
    api_key: str = "",
    model: str = "mercury-2",
    timeout: float = 30.0,
) -> ArgumentGraph | DecompositionError:
    """Decompose a natural-language argument intent into a Toulmin graph.

    Uses constrained prompting: the LLM fills a rigid JSON template with
    forced enum choices. The schema is then parsed and validated structurally
    before any logic checking.

    Args:
        intent: The argument the agent wants to make, in natural language.
        api_base: API endpoint for the LLM.
        api_key: Bearer token for authentication.
        model: Model identifier.
        timeout: HTTP timeout in seconds.

    Returns:
        ArgumentGraph on success, DecompositionError on failure.
    """
    messages = [
        {"role": "system", "content": _DECOMPOSITION_SYSTEM},
        {"role": "user", "content": f"Decompose this argument:\n\n{intent}"},
    ]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{api_base}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.5,  # low but not zero for Mercury's range
                    "reasoning_effort": "medium",
                },
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Decomposition API error: status=%d body=%s", e.response.status_code, e.response.text[:200])
        return DecompositionError(message=f"API returned {e.response.status_code}", raw_response=e.response.text[:500])
    except httpx.TimeoutException:
        logger.error("Decomposition API timeout after %.1fs", timeout)
        return DecompositionError(message=f"API timeout after {timeout}s")
    except httpx.HTTPError as e:
        logger.error("Decomposition API transport error: %s", e)
        return DecompositionError(message=str(e))

    # Extract response text
    try:
        body = resp.json()
        raw_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error("Failed to extract response content: %s", e)
        return DecompositionError(message=f"Malformed API response: {e}", raw_response=resp.text[:500])

    # Parse JSON from response (strip markdown fences if model adds them)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` wrapping
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse decomposition JSON: %s | raw: %s", e, cleaned[:200])
        return DecompositionError(message=f"Invalid JSON from model: {e}", raw_response=cleaned[:500])

    # Build the ArgumentGraph from parsed structure
    return _build_graph(parsed, cleaned)


def _build_graph(parsed: dict[str, object], raw: str) -> ArgumentGraph | DecompositionError:
    """Convert parsed JSON into a typed ArgumentGraph.

    Handles "MISSING" values and enum validation gracefully.
    """
    try:
        # Claim
        claim_text = str(parsed.get("claim_text", "MISSING"))
        if claim_text == "MISSING":
            return DecompositionError(message="Model could not identify a claim", raw_response=raw[:500])

        claim = Claim(
            text=claim_text,
            scope=_parse_enum(ClaimScope, parsed.get("claim_scope"), ClaimScope.PARTICULAR),
            confidence=_parse_enum(Confidence, parsed.get("claim_confidence"), Confidence.PLAUSIBLE),
        )

        # Data
        raw_data = parsed.get("data", [])
        if not isinstance(raw_data, list) or not raw_data:
            return DecompositionError(message="No evidence data provided", raw_response=raw[:500])

        data: list[Datum] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", ""))
            if text and text != "MISSING":
                data.append(Datum(
                    text=text,
                    source=_parse_enum(DataSource, item.get("source"), DataSource.INFERRED),
                    strength=_parse_enum(DataStrength, item.get("strength"), DataStrength.MODERATE),
                ))

        if not data:
            return DecompositionError(message="All evidence entries were MISSING", raw_response=raw[:500])

        # Warrant
        warrant_text = str(parsed.get("warrant_text", "MISSING"))
        if warrant_text == "MISSING":
            return DecompositionError(message="Model could not identify a warrant", raw_response=raw[:500])

        warrant = Warrant(
            text=warrant_text,
            inference_type=_parse_enum(InferenceType, parsed.get("warrant_type"), InferenceType.INDUCTIVE),
            explicit=True,
            formalization=_str_or_none(parsed.get("warrant_formalization")),
        )

        # Backing (optional)
        backing_text = str(parsed.get("backing_text", "MISSING"))
        backing = None
        if backing_text != "MISSING":
            backing = Backing(
                text=backing_text,
                source=_parse_enum(BackingSource, parsed.get("backing_source"), BackingSource.DOMAIN_EXPERTISE),
            )

        # Qualifier (optional)
        qualifier_text = str(parsed.get("qualifier_text", "MISSING"))
        qualifier = None
        if qualifier_text != "MISSING":
            qualifier = Qualifier(
                text=qualifier_text,
                strength=_parse_enum(Confidence, parsed.get("qualifier_strength"), Confidence.PLAUSIBLE),
                calibrated=False,
            )

        # Rebuttals (optional)
        raw_rebuttals = parsed.get("rebuttals", [])
        rebuttals: list[Rebuttal] = []
        if isinstance(raw_rebuttals, list):
            for item in raw_rebuttals:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", ""))
                if text and text != "MISSING":
                    rebuttals.append(Rebuttal(
                        text=text,
                        severity=_parse_enum(RebuttalSeverity, item.get("severity"), RebuttalSeverity.SIGNIFICANT),
                        addressed=item.get("response") is not None and str(item.get("response")) != "MISSING",
                        response=_str_or_none(item.get("response")),
                    ))

        graph = ArgumentGraph(
            claim=claim,
            data=data,
            warrant=warrant,
            backing=backing,
            qualifier=qualifier,
            rebuttals=rebuttals,
        )

        logger.info(
            "Decomposition successful: graph=%s claim_scope=%s data_count=%d rebuttal_count=%d",
            graph.id, claim.scope.value, len(data), len(rebuttals),
        )

        return graph

    except Exception as e:
        logger.error("Failed to build ArgumentGraph: %s", e, exc_info=True)
        return DecompositionError(message=f"Graph construction failed: {e}", raw_response=raw[:500])


def _parse_enum[E](enum_cls: type[E], value: object, default: E) -> E:
    """Safely parse an enum value with fallback."""
    if value is None or str(value) == "MISSING":
        return default
    try:
        return enum_cls(str(value).lower())  # type: ignore[call-arg]
    except (ValueError, KeyError):
        return default


def _str_or_none(value: object) -> str | None:
    """Convert to string or None if MISSING/null."""
    if value is None:
        return None
    s = str(value)
    return None if s == "MISSING" else s

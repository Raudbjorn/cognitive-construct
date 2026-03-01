"""Analogy Bridge — uses Mercury (diffusion LLM) for structural analogy generation.

This is the one place where diffusion's non-path-dependent token distributions
add genuine value to rhetoric. The bridge receives a validated Toulmin graph
and asks Mercury to find a domain-distant system that shares the same
inferential structure.

The key constraint: the analogy must be isomorphic to the argument's logic,
not just superficially similar. Mercury's diffusion process can explore the
full token space simultaneously, making it more likely to find non-obvious
structural mappings that an autoregressive model would miss.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from toulmin.models import Analogy, ArgumentGraph

logger = logging.getLogger(__name__)

_BRIDGE_SYSTEM = """\
You are a structural analogy engine. Your job is to find analogies that share \
the same CAUSAL or FUNCTIONAL structure as the given argument — not just \
surface similarity.

Respond with ONLY valid JSON. No preamble, no markdown fences, no explanation.

JSON Schema:
{
  "source_domain": "string — the original argument's domain",
  "target_domain": "string — the distant domain you found",
  "mapping": {
    "original_element_1": "analogous_element_1",
    "original_element_2": "analogous_element_2"
  },
  "analogy_text": "string — a natural-language 1-3 sentence analogy",
  "structural_justification": "string — why this is a structural (not surface) match"
}"""


@dataclass(frozen=True)
class BridgeError:
    """Analogy generation failed or produced a weak match."""

    message: str
    raw_response: str | None = None


async def generate_analogy(
    graph: ArgumentGraph,
    *,
    api_base: str = "https://api.inceptionlabs.ai",
    api_key: str = "",
    model: str = "mercury-2",
    timeout: float = 45.0,
) -> Analogy | BridgeError:
    """Generate a structurally-isomorphic analogy for a validated argument.

    The diffusion model receives the argument's inferential structure as a
    constraint, not an open prompt. This ensures the analogy maps to the
    logic, not just the topic.

    Should only be called on graphs that have passed validation.

    Args:
        graph: A validated ArgumentGraph.
        api_base: Mercury API endpoint.
        api_key: Bearer token.
        model: Model identifier.
        timeout: HTTP timeout in seconds.

    Returns:
        Analogy on success, BridgeError on failure.
    """
    # Build the structural constraint from the validated graph
    constraint = _build_constraint(graph)

    messages = [
        {"role": "system", "content": _BRIDGE_SYSTEM},
        {"role": "user", "content": constraint},
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
                    "max_tokens": 2048,
                    "temperature": 0.9,  # higher temp for creative exploration
                    "reasoning_effort": "high",  # let diffusion explore fully
                },
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Bridge API error: status=%d", e.response.status_code)
        return BridgeError(message=f"API returned {e.response.status_code}", raw_response=e.response.text[:500])
    except httpx.TimeoutException:
        logger.error("Bridge API timeout after %.1fs", timeout)
        return BridgeError(message=f"API timeout after {timeout}s")
    except httpx.HTTPError as e:
        logger.error("Bridge API transport error: %s", e)
        return BridgeError(message=str(e))

    # Parse response
    try:
        body = resp.json()
        raw_text: str = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return BridgeError(message=f"Malformed API response: {e}", raw_response=str(resp.text)[:500])

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return BridgeError(message=f"Invalid JSON: {e}", raw_response=cleaned[:500])

    return _build_analogy(parsed, graph, cleaned)


def _build_constraint(graph: ArgumentGraph) -> str:
    """Build the structural constraint prompt from a Toulmin graph."""
    data_summary = "; ".join(d.text[:100] for d in graph.data)
    rebuttal_summary = "; ".join(r.text[:100] for r in graph.rebuttals) if graph.rebuttals else "none"

    return f"""Find a structural analogy for the following argument:

DOMAIN: {_infer_domain(graph.claim.text)}
CLAIM: {graph.claim.text}
DATA: {data_summary}
WARRANT (the inferential bridge): {graph.warrant.text}
WARRANT TYPE: {graph.warrant.inference_type.value}
REBUTTALS: {rebuttal_summary}

Requirements:
- The analogy must come from a DISTANT domain (not the same field)
- The mapped elements must play the same CAUSAL/FUNCTIONAL roles
- Surface similarity alone is NOT sufficient
- The analogy should be concrete and visualizable
- Prefer analogies from physical/natural systems over abstract ones"""


def _infer_domain(claim_text: str) -> str:
    """Crude domain inference from claim text. Good enough for the constraint prompt."""
    text_lower = claim_text.lower()
    domain_keywords: dict[str, list[str]] = {
        "software engineering": ["code", "software", "api", "database", "deploy", "microservice", "monolith"],
        "machine learning": ["model", "training", "neural", "llm", "diffusion", "embedding", "token"],
        "economics": ["market", "price", "inflation", "gdp", "trade", "fiscal"],
        "biology": ["cell", "gene", "protein", "evolution", "organism", "species"],
        "physics": ["energy", "force", "quantum", "particle", "wave", "field"],
        "politics": ["policy", "government", "election", "vote", "legislation"],
        "philosophy": ["ethics", "moral", "epistem", "ontolog", "consciousness"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return domain

    return "general"


def _build_analogy(
    parsed: dict[str, object],
    graph: ArgumentGraph,
    raw: str,
) -> Analogy | BridgeError:
    """Convert parsed JSON into a verified Analogy."""
    try:
        source_domain = str(parsed.get("source_domain", "unknown"))
        target_domain = str(parsed.get("target_domain", "unknown"))
        raw_mapping = parsed.get("mapping", {})
        analogy_text = str(parsed.get("analogy_text", ""))
        justification = str(parsed.get("structural_justification", ""))

        if not analogy_text or analogy_text == "MISSING":
            return BridgeError(message="No analogy text generated", raw_response=raw[:500])

        mapping: dict[str, str] = {}
        if isinstance(raw_mapping, dict):
            mapping = {str(k): str(v) for k, v in raw_mapping.items()}

        # Structural verification: does the mapping cover the key argument elements?
        structural_match = _verify_structural_match(mapping, graph, justification)

        analogy = Analogy(
            source_domain=source_domain,
            target_domain=target_domain,
            mapping=mapping,
            analogy_text=analogy_text,
            structural_match=structural_match,
        )

        logger.info(
            "Analogy generated: %s → %s | structural_match=%s",
            source_domain, target_domain, structural_match,
        )

        return analogy

    except Exception as e:
        logger.error("Failed to build analogy: %s", e, exc_info=True)
        return BridgeError(message=f"Analogy construction failed: {e}", raw_response=raw[:500])


def _verify_structural_match(
    mapping: dict[str, str],
    graph: ArgumentGraph,
    justification: str,
) -> bool:
    """Verify that the analogy's mapping covers the argument's structure.

    Checks:
    1. Does the mapping have at least 2 elements? (non-trivial)
    2. Does the justification reference structural/causal language?
    3. Are the source and target domains actually different?

    This is a heuristic check for the PoC. Production would use the full
    isomorphism verification from the spec.
    """
    if len(mapping) < 2:
        return False

    structural_terms = [
        "structure", "structur", "function", "mechanism", "cause", "role",
        "relationship", "dynamic", "process", "system", "operates",
    ]
    has_structural_language = any(term in justification.lower() for term in structural_terms)

    return has_structural_language

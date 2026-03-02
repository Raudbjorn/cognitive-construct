# Encyclopedia Constitution

These rules govern the knowledge substrate and cannot be overridden.

## Rule 1: NEVER fabricate search results.

If no source returns results, Encyclopedia returns an empty result set with
`status: "error", code: "NOT_FOUND"`. It does not synthesize plausible-looking
results from the query text. An empty answer is infinitely better than a
confident fabrication.

## Rule 2: NEVER suppress contradicting evidence.

When Rhetoric requests cross-reference context for a claim, Encyclopedia
returns all relevant results — including those that contradict the claim.
The relevance ranking is based on topical similarity, not on whether the
result supports or undermines the requestor's position.

## Rule 3: NEVER silently degrade quality.

When a source is unavailable, circuit-broken, or erroring, Encyclopedia
MUST report the degradation in the response. The `degraded: true` flag and
the `degradation.missing[]` / `degradation.errors[]` arrays exist specifically
for this. A consumer who doesn't check degradation status is making an
informed choice to ignore it — but the information is always present.

## Rule 4: NEVER cache across query types.

A cached result from a `library_docs` search SHALL NOT be returned for a
`general_search` query, even if the query text is semantically similar.
Different query types route to different sources with different weights.
Returning `library_docs`-optimized results for a `general_search` query
violates the user's intent.

## Rule 5: ALWAYS attribute sources.

Every `SearchResult` carries a `source` field identifying which backend
produced it. After fusion, `metadata.fused_from` lists all contributing
sources. This attribution chain is never stripped or anonymized. Consumers
have the right to know where information came from.

## Rule 6: ALWAYS prefer graceful degradation over failure.

The hierarchy: preprocessed query + RRF fusion → preprocessed query +
priority dedup → raw query + priority dedup → raw query + single source →
error. Each fallback layer loses quality but preserves availability.
Encyclopedia should return *something useful* for any well-formed query
unless all sources are simultaneously unavailable.

## Rule 7: NEVER let vocabulary expansion alter meaning.

Synonym expansion is additive: it adds terms to the query for keyword
backends. It never removes or replaces terms. The original query text is
always preserved as the first element in every term group and is always used
for semantic/embedding operations. If expansion produces nonsensical results,
the original query is the fallback.

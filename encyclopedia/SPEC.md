# Encyclopedia Technical Specification

> The agent that knows nothing knows nothing worth saying.

**Version:** 0.2.0-draft
**Status:** Partial Implementation (Phases 1–2 complete, Phases 3–6 design)
**Depends on:** `shared.embeddings`, `shared.fusion`, `shared.vocabulary`, `shared.query_pipeline`, `shared.events`, `shared.membrane`, `shared.synergies`, `shared.feedback`, `shared.mcp_client`

---

## 1. Problem Statement

### 1.1 The Raw Query Problem

Encyclopedia is the knowledge substrate of the Cognitive Construct. Every
other skill — Rhetoric, Volition, Sparring Partner, Inland Empire — depends
on it for facts. When Encyclopedia returns bad results, every downstream
consumer builds on sand.

The original implementation (`encyclopedia.py` v2.1.0) had a structural flaw:
it sent the user's raw query verbatim to every backend source. A query like
"k8s auth middleware" was forwarded unchanged to Context7, Exa, and Perplexity
— three backends with fundamentally different query semantics. Context7 wants
library-scoped lookups. Exa wants keyword-rich web queries. Perplexity wants
natural-language questions. Treating them identically guaranteed that at least
two of three would receive a suboptimal query.

This is the equivalent of asking the same question in the same tone to a
librarian, a search engine, and a colleague. One of them might understand
you. The other two are working with the wrong input.

### 1.2 The Priority Dedup Problem

After results returned, deduplication was priority-ordered:

```python
SOURCE_PRIORITY = {
    "context7": 100,
    "exa": 80,
    "perplexity": 70,
    ...
}
```

When two sources returned the same document, the higher-priority source's
version was kept and the lower-priority source's version was discarded.
This meant Context7 *always* beat Exa, regardless of which source actually
returned the better result for *this specific query*.

Worse: cross-source agreement — the strongest signal that a result is
genuinely relevant — was treated as noise to be eliminated. If both Context7
and Exa returned the same page, that convergence was information being
destroyed, not redundancy being cleaned up.

### 1.3 What This Specification Addresses

This document specifies Encyclopedia's transformation from a
broadcast-and-dedup system into an intelligent retrieval pipeline:

- **Query preprocessing** that adapts queries to each source's strengths
  (synonym expansion for keyword backends, clean queries for semantic backends)
- **Reciprocal Rank Fusion** that treats cross-source agreement as a ranking
  signal instead of throwing it away
- **Domain vocabulary** that captures the terminology of the query space
  (dev/programming, with swappable JSON vocabularies for other domains)
- **Source health tracking** with circuit breakers and graceful degradation
- **Semantic caching** that prevents redundant queries for near-duplicate inputs
- **Source quality profiling** that learns which sources perform well for
  which query types over time

### 1.4 What This Specification Does Not Address

- The internal implementation of backend clients (Context7, Exa, Perplexity)
- The Claude Code SKILL.md interface and CLI argument parsing
- Real-time streaming of search results to the user
- Multi-user concurrency (Encyclopedia operates in single-user CLI context)
- The diffusion/creativity pipeline (that's Inner-Monologue)

---

## 2. Requirements

### 2.1 User Stories

#### US-1: Query Preprocessing

**As the** orchestrating agent
**I want to** have my queries automatically corrected, expanded, and adapted
before they reach backend sources
**So that** each source receives the query form it handles best

EARS acceptance:
> WHEN a query contains a known misspelling (e.g., "kuberntes"),
> the system SHALL correct it before dispatching to any source.

> WHEN a query contains an expandable abbreviation (e.g., "k8s"),
> the system SHALL expand it for keyword-oriented sources (Exa, Kagi, SearXNG)
> AND SHALL use only the corrected form for semantic sources (Perplexity, Context7).

> WHEN query preprocessing fails for any reason,
> the system SHALL fall back to the raw query without error.

#### US-2: Intelligent Result Fusion

**As the** orchestrating agent
**I want to** rank results using cross-source agreement as a signal
**So that** documents confirmed by multiple sources rank higher than
documents from a single high-priority source

EARS acceptance:
> WHEN multiple sources return the same document,
> the system SHALL boost its ranking proportionally to source count
> using Reciprocal Rank Fusion with per-query-type source weights.

> WHEN only one source returns results (degraded mode),
> the system SHALL fall back to single-source ranking by position.

> WHEN the fusion engine is unavailable,
> the system SHALL fall back to priority-based deduplication.

#### US-3: Source Health and Degradation

**As the** orchestrating agent
**I want to** know when sources are degraded, and have the system
adapt rather than silently returning worse results
**So that** I can trust the quality assessment attached to results

EARS acceptance:
> WHEN a source returns errors for 3+ consecutive queries,
> the system SHALL circuit-break that source for a configurable
> cooldown period (default 60 seconds).

> WHEN a source is circuit-broken,
> the system SHALL exclude it from query dispatch AND record the
> event via `shared.events.create_event(ENCYCLOPEDIA_SOURCE_DEGRADED)`.

> WHEN a circuit-broken source's cooldown expires,
> the system SHALL probe it with the next query AND emit
> `ENCYCLOPEDIA_SOURCE_RESTORED` on success.

> WHEN all sources for a query type are unavailable,
> the system SHALL attempt the query against fallback sources
> before returning NOT_FOUND.

#### US-4: Semantic Cache

**As the** orchestrating agent
**I want to** avoid re-querying backends for semantically similar queries
issued within a short time window
**So that** repeated or paraphrased questions are fast and don't waste
API quota

EARS acceptance:
> WHEN a new query's embedding similarity to a cached query exceeds
> the cache threshold (default 0.92) AND the cached result is younger
> than the cache TTL (default 1 hour),
> the system SHALL return the cached result with `cached: true` metadata.

> WHEN returning cached results,
> the system SHALL include `ENCYCLOPEDIA_CACHE_HIT` in the event stream.

> WHEN cache lookup fails or is disabled,
> the system SHALL proceed with a normal query without error.

#### US-5: Cross-Reference Interface (Rhetoric Integration)

**As the** Rhetoric skill
**I want to** query Encyclopedia for evidence that contradicts or supports
a specific claim
**So that** I can detect cherry-picking and incomplete rebuttals

EARS acceptance:
> WHEN Rhetoric sends a `context_lookup` message via `shared.synergies`,
> the system SHALL return the top-3 most relevant results for the claim
> with relevance scores, source attribution, and contradiction indicators.

> WHEN no relevant evidence exists,
> the system SHALL return an empty result set (not an error).

#### US-6: Volition Fallback Search

**As the** Volition skill
**I want to** trigger an Encyclopedia search when the primary action
handler lacks context
**So that** action plans are informed by the best available knowledge

EARS acceptance:
> WHEN Volition emits `encyclopedia.cache.miss` via the membrane,
> the system SHALL execute a search with the payload query
> and return results through the synergy bus.

#### US-7: Source Quality Profiling

**As a** system that improves over time
**I want to** track which sources produce useful results for which query types
**So that** source weights can be refined from empirical data

EARS acceptance:
> WHEN a search completes, the system SHALL record per-source
> result-count, latency, and (when available) downstream feedback scores
> in the source quality log.

> WHEN `ENCYCLOPEDIA_ADAPTIVE_WEIGHTS` is enabled,
> the system SHALL adjust RRF source weights based on the trailing
> quality profile (sliding window, last 200 queries per type).

### 2.2 Scope

**In scope:**
- Query preprocessing pipeline (normalize → correct → expand → classify)
- Domain vocabulary management (JSON-loadable, multi-domain)
- RRF-based result fusion with per-query-type source weights
- Source health monitoring and circuit breaking
- Semantic caching with embedding similarity
- Cross-reference interface for Rhetoric
- Fallback search interface for Volition
- Source quality logging and adaptive weight adjustment
- Graceful degradation at every layer

**Out of scope (this version):**
- Adding new backend sources (Kagi, SearXNG, CodeGraphContext remain optional/stub)
- Streaming results to the user during query execution
- Multi-query session context ("search again but exclude React results")
- User-configurable source weights via CLI flags
- Embedding model fine-tuning on query/result pairs

### 2.3 Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Latency | Query preprocessing SHALL complete in < 5ms (pure string ops, no I/O) |
| NFR-2 | Latency | RRF fusion SHALL complete in < 10ms for ≤100 results across ≤7 sources |
| NFR-3 | Latency | Semantic cache lookup SHALL complete in < 15ms (single embedding + cosine) |
| NFR-4 | Latency | Total overhead (preprocessing + fusion + cache check) SHALL be < 30ms — negligible vs backend network I/O |
| NFR-5 | Availability | If `shared.query_pipeline` import fails, the system SHALL fall back to raw query dispatch |
| NFR-6 | Availability | If `shared.fusion` import fails, the system SHALL fall back to priority-based dedup |
| NFR-7 | Availability | If all sources for a query type are unavailable, the system SHALL attempt cross-type fallback before failing |
| NFR-8 | Determinism | Given identical query, vocabulary, and source results, fusion scores SHALL be identical |
| NFR-9 | Memory | Vocabulary + embedding model SHALL not exceed 25MB resident (vocabulary ~1MB, Model2Vec ~8MB, cache ~16MB) |
| NFR-10 | Auditability | Every search SHALL be loggable with query analysis, source dispatch, and fusion scores |

---

## 3. Design

### 3.1 Architecture Overview

```
                      user query
                          │
                          ▼
                ┌─────────────────────┐
                │   Query Pipeline     │
                │ normalize → correct  │
                │ → expand → classify  │
                └──────────┬──────────┘
                           │
                    ProcessedQuery
                   ╱       │       ╲
           keyword_query   │   semantic_query
              │            │         │
              ▼            │         ▼
        ┌─────────┐       │    ┌──────────┐
        │ Exa     │       │    │ Perplexity│
        │ Kagi    │       │    │ Context7  │
        │ SearXNG │       │    └─────┬─────┘
        └────┬────┘       │          │
             │            │          │
             ▼            │          ▼
        ┌─────────────────┴──────────────┐
        │          RRF Fusion             │
        │  per-source weights × rank      │
        │  cross-source boost             │
        └──────────────┬─────────────────┘
                       │
                       ▼
                 FusedResult[]
                       │
                       ▼
                  response JSON
                  + query_analysis
```

### 3.2 Data Structures

#### 3.2.1 SearchResult (existing, extended)

The core result type returned by every backend source adapter.

```python
@dataclass
class SearchResult:
    title: str
    content: str
    url: str = ""
    source: str = ""
    relevance: float = 0.0
    metadata: dict = field(default_factory=dict)

    # --- Added by fusion layer ---
    # metadata["fused_from"]: list[str]  — sources that returned this result
    # metadata["source_count"]: int      — number of agreeing sources
    # metadata["query_analysis"]: dict   — preprocessing diagnostics (on first result only)
```

**Design decision:** Extending `metadata` rather than adding new fields preserves
backward compatibility. Consumers that don't know about fusion ignore the extra
keys. Consumers that do (Rhetoric, Volition) can read them.

#### 3.2.2 ProcessedQuery (from `shared.query_pipeline`)

The output of the preprocessing pipeline. Carries differential query forms.

```python
@dataclass
class ProcessedQuery:
    original: str              # Raw user input
    corrected: str             # After typo correction
    corrections: list[Correction]  # What was corrected and why
    expanded: ExpandedQuery    # Term groups with synonyms
    text_for_embedding: str    # Corrected only — no synonym noise
    query_type: str            # library_docs | general_search | code_context | repository
    repo_hint: str | None      # Extracted repo:owner/name
    cleaned_query: str         # After prefix/hint removal
    suggestions: list[str]     # Human-readable tips ("k8s = kubernetes")
```

**The differential query insight** (ported from TTTTRPS `pipeline.rs:114`):
keyword sources receive `expanded.expanded_text` (synonyms help BM25 recall),
while semantic/AI sources receive `text_for_embedding` (synonyms degrade
embedding quality by adding distributional noise). This single decision is
responsible for the largest quality improvement in the pipeline.

#### 3.2.3 FusedResult (from `shared.fusion`)

The output of RRF fusion. Replaces the old priority-deduplicated list.

```python
@dataclass
class FusedResult(Generic[T]):
    item: T                          # The original SearchResult
    score: float                     # Normalized RRF score [0, 1]
    source_ranks: dict[str, int]     # Per-source rank positions
    source_count: int                # Number of sources that returned this
    sources: list[str]               # Which sources (derived from source_ranks)
```

**Design decision:** `source_count` is the key new signal. A result with
`source_count=3` and moderate per-source ranking is almost certainly more
relevant than a result with `source_count=1` and top ranking from a single
source. RRF captures this naturally through additive scoring.

#### 3.2.4 SourceHealthState (new, Phase 3)

Per-source circuit breaker state.

```json
{
  "source": "context7",
  "status": "healthy | degraded | circuit_open",
  "consecutive_failures": 0,
  "last_failure_at": null,
  "last_success_at": "2026-03-01T14:00:00Z",
  "circuit_opened_at": null,
  "cooldown_seconds": 60,
  "total_queries": 412,
  "total_failures": 7,
  "avg_latency_ms": 340
}
```

State machine:
```
healthy ──[3 consecutive failures]──> circuit_open
circuit_open ──[cooldown expires]──> degraded (probe mode)
degraded ──[probe succeeds]──> healthy
degraded ──[probe fails]──> circuit_open (reset cooldown)
```

#### 3.2.5 SemanticCacheEntry (new, Phase 4)

Cached result keyed by query embedding.

```json
{
  "query": "kubernetes auth middleware",
  "query_embedding_hash": "sha256:a1b2c3...",
  "query_type": "library_docs",
  "results": [ "...SearchResult[]..." ],
  "sources_used": ["context7", "exa"],
  "created_at": "2026-03-01T14:00:00Z",
  "ttl_seconds": 3600,
  "hit_count": 0
}
```

Cache lookup uses cosine similarity between the new query's embedding and
stored embeddings. Threshold of 0.92 was chosen empirically: high enough that
"kubernetes auth" and "k8s authentication middleware" are considered the same
query, low enough that "kubernetes auth" and "kubernetes networking" are not.

#### 3.2.6 SourceQualityProfile (new, Phase 5)

Rolling quality metrics per source per query type.

```json
{
  "source": "exa",
  "query_type": "library_docs",
  "window_size": 200,
  "metrics": {
    "avg_result_count": 4.2,
    "avg_latency_ms": 280,
    "avg_fusion_rank": 2.1,
    "feedback_score": 0.72,
    "timeout_rate": 0.02,
    "empty_rate": 0.05
  },
  "recommended_weight_adjustment": +0.05,
  "last_updated": "2026-03-01T14:00:00Z"
}
```

### 3.3 Query Preprocessing Engine

**Status:** ✅ Implemented (`shared/query_pipeline.py`, `shared/vocabulary.py`)

The pipeline processes queries in five stages:

| Stage | Function | Example |
|-------|----------|---------|
| Normalize | Lowercase, collapse whitespace, strip | `"  K8S  Auth "` → `"k8s auth"` |
| Extract metadata | Repo hints, type prefixes | `"repo:foo/bar doc: k8s auth"` → hint=`foo/bar`, type=`library_docs`, query=`"k8s auth"` |
| Typo correct | Static table + Levenshtein fallback | `"kuberntes"` → `"kubernetes"` |
| Synonym expand | Multi-way + one-way vocabulary | `"k8s"` → `["k8s", "kubernetes"]` |
| Classify | URL/code/time/library patterns | `"k8s auth"` → `library_docs` |

The vocabulary is loaded from `shared/vocabularies/dev_programming.json`:
~80 multi-way synonym groups, ~40 common misspellings, ~150 known terms.
The JSON format makes vocabulary swappable without code changes — a film/location
vocabulary for Massif, a legal vocabulary for contracts, etc.

**Inflection awareness:** The typo corrector recognizes common English inflections
(plurals, -ed, -ing) of known terms as valid. "hooks" is not corrected to "hook"
even though "hook" is a known term. This prevents the corrector from aggressively
"fixing" grammatically correct queries.

### 3.4 Result Fusion Engine

**Status:** ✅ Implemented (`shared/fusion.py`)

Reciprocal Rank Fusion replaces priority-based deduplication. The RRF score
for a document `d` across `n` source rankings:

```
score(d) = Σᵢ wᵢ / (k + rankᵢ(d))
```

Where `wᵢ` is the source weight for source `i`, `k` is the smoothing
constant (default 60), and `rankᵢ(d)` is the document's position in source
`i`'s results (absent → excluded from sum).

**Per-query-type source weights:**

| Query Type | Context7 | Exa | Perplexity | Kagi | SearXNG | mcp-git-ingest | CodeGraph |
|-----------|----------|-----|------------|------|---------|----------------|-----------|
| library_docs | 0.40 | 0.30 | 0.20 | 0.10 | — | — | — |
| general_search | — | 0.35 | 0.35 | 0.20 | 0.10 | — | — |
| code_context | — | 0.20 | 0.15 | — | — | 0.40 | 0.25 |
| repository | — | 0.20 | — | — | — | 0.70 | 0.10 |

**Weight rationale:** Context7 dominates `library_docs` because it has
curated, version-specific documentation — the highest-signal source for
that query type. For `general_search`, Exa and Perplexity are balanced
because neither consistently outperforms the other. `mcp-git-ingest` dominates
`code_context` and `repository` because it has direct repository access.

**Fallback:** When weights are unavailable for a query type (unknown or custom),
the engine falls back to equal weights across all responding sources. This is
always safe — RRF with equal weights is equivalent to Borda count.

### 3.5 Source Health Monitor

**Status:** 🔲 Design (Phase 3)

Each source is wrapped in a `SourceHealth` circuit breaker that tracks
consecutive failures and enforces cooldown periods.

```python
class SourceHealth:
    """Circuit breaker for a single search backend."""

    def __init__(self, source: str, failure_threshold: int = 3, cooldown: float = 60.0):
        self.source = source
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None

    def is_available(self) -> bool:
        if self._consecutive_failures < self.failure_threshold:
            return True
        if self._circuit_opened_at is None:
            return False
        # Allow probe after cooldown
        return (time.monotonic() - self._circuit_opened_at) >= self.cooldown

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._circuit_opened_at = time.monotonic()
```

**Design decision: Conservative thresholds.** 3 consecutive failures before
circuit-break, 60-second cooldown. This is conservative because false
circuit-breaks (marking a healthy source as down) are worse than slow
circuit-breaks (continuing to try a failing source for a few extra queries).
A single network hiccup does not disable a source.

### 3.6 Semantic Cache

**Status:** 🔲 Design (Phase 4)

The current file-based cache uses exact query-string matching (`get_cache_key()`
hashes the query + source). This misses paraphrases: "kubernetes authentication"
and "k8s auth middleware" are different cache keys despite being semantically
near-identical queries.

The semantic cache replaces string hashing with embedding similarity:

```python
async def check_semantic_cache(query: str, query_type: str) -> list[SearchResult] | None:
    query_embedding = encode(query)  # shared.embeddings.encode()
    for entry in _cache_entries:
        if entry.query_type != query_type:
            continue
        if entry.is_expired():
            continue
        similarity = cosine_similarity(query_embedding, entry.embedding)
        if similarity >= CACHE_SIMILARITY_THRESHOLD:
            entry.hit_count += 1
            return entry.results
    return None
```

**Design decision: Cache by query type.** A `library_docs` cache entry should
not be returned for a `general_search` query with similar text. The source
routing is different, so the cached results would be from the wrong backends.

**Design decision: In-memory with persistence.** The cache lives in memory
during a session and is persisted to `~/.encyclopedia/cache/` as JSONL on
graceful shutdown. This avoids disk I/O on the hot path. Cache size is bounded
at 500 entries (LRU eviction), ~16MB worst case.

### 3.7 Source Quality Profiling

**Status:** 🔲 Design (Phase 5)

The source quality profiler tracks rolling metrics per source per query type.
This closes the feedback loop: instead of static source weights hardcoded in
`ENCYCLOPEDIA_WEIGHTS`, weights adapt to empirical performance.

Signals collected per query:
- **Result count**: Sources that consistently return 0 results for a query type
  should have their weight reduced.
- **Fusion rank**: If a source's results consistently rank low after fusion
  (other sources' results are preferred), its weight should decrease.
- **Latency**: Sources that consistently timeout should have their weight
  reduced to deprioritize slow backends.
- **Downstream feedback**: When `shared.feedback.FeedbackCollector` receives
  explicit signals (user marked a result as useful/not useful), the originating
  source gets credit/blame.

Weight adjustment is conservative: ±5% per profiling window, capped at ±20%
total drift from the static baseline. This prevents a feedback spiral where
a temporarily degraded source gets permanently deprioritized.

**Feature flag:** `ENCYCLOPEDIA_ADAPTIVE_WEIGHTS` (default: disabled). Static
weights are the safe default. Adaptive weights require sufficient query volume
(200+ per type) to produce stable estimates.

### 3.8 Integration Points

Encyclopedia integrates with the Cognitive Construct through three channels:

#### Event Bus (`shared.events`)

Encyclopedia emits six event types defined in `shared/events.py`:

| Event | Trigger | Consumers |
|-------|---------|-----------|
| `ENCYCLOPEDIA_SEARCH_STARTED` | Query dispatch begins | Audit log |
| `ENCYCLOPEDIA_SEARCH_COMPLETED` | Results fused and returned | Inland Empire (memory), Feedback Collector |
| `ENCYCLOPEDIA_CACHE_HIT` | Semantic cache match found | Audit log, metrics |
| `ENCYCLOPEDIA_CACHE_MISS` | No cache match, full search required | Volition (fallback trigger) |
| `ENCYCLOPEDIA_SOURCE_DEGRADED` | Source circuit-broken | Audit log, health dashboard |
| `ENCYCLOPEDIA_SOURCE_RESTORED` | Source recovered from circuit-break | Audit log, health dashboard |

#### Membrane (`shared.membrane`)

Encyclopedia's membrane (`shared/membrane.py:create_encyclopedia_membrane()`)
absorbs events from other skills and emits search events:

```
absorbs:
  - rhetoric.thought.recorded (when needs_research=true)
    → triggers a search for supporting/contradicting evidence
  - volition.action.failed (when handler=web_search)
    → Volition's web search failed, Encyclopedia provides fallback

emits:
  - encyclopedia.search.completed → Inland Empire, Feedback Collector
  - encyclopedia.source.degraded → Audit, Volition (route around)
  - encyclopedia.source.restored → Volition (restore routes)
```

#### Synergy Bus (`shared.synergies`)

Two synergy functions connect Encyclopedia to other skills:

- **`rhetoric_request_context(query)`**: Rhetoric calls this to cross-reference
  claims against Encyclopedia's knowledge. Returns top-3 results with relevance
  scores. Used by Rhetoric's Pass 3 (Cross-Reference Integrity).

- **`encyclopedia_context_handler(message)`**: Handles incoming `CONTEXT_LOOKUP`
  messages from any skill. Executes a search and returns results through the
  synergy response channel.

### 3.9 Information Flow

The complete lifecycle of a single search query:

```
 1. USER QUERY (or skill request via synergy bus)
    │
    ▼
 2. SEMANTIC CACHE CHECK
    │ Embed query → compare against cached embeddings
    │ Hit? → return cached results (skip steps 3–8)
    │ Miss? → continue
    ▼
 3. QUERY PREPROCESSING (shared.query_pipeline)
    │ normalize → extract metadata → typo correct → expand → classify
    │ Output: ProcessedQuery with keyword_query + semantic_query
    ▼
 4. SOURCE ROUTING
    │ query_type → SOURCE_ROUTING table → candidate sources
    │ Filter by: credentials available, feature flags enabled, circuit not open
    │ Output: target_sources[] with per-source query variant
    ▼
 5. PARALLEL DISPATCH
    │ Fan out to target sources with per-source timeout
    │ keyword_query → Exa, Kagi, SearXNG
    │ semantic_query → Perplexity, Context7
    │ repo_hint → mcp-git-ingest
    ▼
 6. RESULT COLLECTION
    │ Gather results, record successes/failures in circuit breakers
    │ Record per-source latency and result count in quality profiler
    ▼
 7. RRF FUSION (shared.fusion)
    │ Per-query-type source weights × reciprocal rank
    │ Cross-source agreement boosts ranking
    │ Output: FusedResult[] sorted by score
    ▼
 8. CACHE STORE
    │ Store results + query embedding in semantic cache
    ▼
 9. EVENT EMISSION
    │ ENCYCLOPEDIA_SEARCH_COMPLETED → event bus
    │ query_analysis + fusion scores → audit log
    ▼
10. RESPONSE
    │ JSON with results, sources_used, query_analysis, degradation info
```

---

## 4. Constitution

These rules govern the knowledge substrate and cannot be overridden.

### Rule 1: NEVER fabricate search results.

If no source returns results, Encyclopedia returns an empty result set with
`status: "error", code: "NOT_FOUND"`. It does not synthesize plausible-looking
results from the query text. An empty answer is infinitely better than a
confident fabrication.

### Rule 2: NEVER suppress contradicting evidence.

When Rhetoric requests cross-reference context for a claim, Encyclopedia
returns all relevant results — including those that contradict the claim.
The relevance ranking is based on topical similarity, not on whether the
result supports or undermines the requestor's position.

### Rule 3: NEVER silently degrade quality.

When a source is unavailable, circuit-broken, or erroring, Encyclopedia
MUST report the degradation in the response. The `degraded: true` flag and
the `degradation.missing[]` / `degradation.errors[]` arrays exist specifically
for this. A consumer that doesn't check degradation status is making an
informed choice to ignore it — but the information is always present.

### Rule 4: NEVER cache across query types.

A cached result from a `library_docs` search SHALL NOT be returned for a
`general_search` query, even if the query text is semantically similar.
Different query types route to different sources with different weights.
Returning `library_docs`-optimized results for a `general_search` query
violates the user's intent.

### Rule 5: ALWAYS attribute sources.

Every `SearchResult` carries a `source` field identifying which backend
produced it. After fusion, `metadata.fused_from` lists all contributing
sources. This attribution chain is never stripped or anonymized. Consumers
have the right to know where information came from.

### Rule 6: ALWAYS prefer graceful degradation over failure.

The hierarchy: preprocessed query + RRF fusion → preprocessed query +
priority dedup → raw query + priority dedup → raw query + single source →
error. Each fallback layer loses quality but preserves availability.
Encyclopedia should return *something useful* for any well-formed query
unless all sources are simultaneously unavailable.

### Rule 7: NEVER let vocabulary expansion alter meaning.

Synonym expansion is additive: it adds terms to the query for keyword
backends. It never removes or replaces terms. The original query text is
always preserved as the first element in every term group and is always used
for semantic/embedding operations. If expansion produces nonsensical results,
the original query is the fallback.

---

## 5. Tasks

### Phase 1: Query Preprocessing Pipeline ✅

**Goal:** Replace raw-query broadcast with differential query dispatch.

| Task | Description | Status | Files |
|------|-------------|--------|-------|
| 1.1 | Implement `Vocabulary` class (multi-way, one-way, typo correction, inflection-aware) | ✅ Done | `shared/vocabulary.py` |
| 1.2 | Create dev/programming domain vocabulary | ✅ Done | `shared/vocabularies/dev_programming.json` |
| 1.3 | Implement `QueryPipeline` (normalize → correct → expand → classify) | ✅ Done | `shared/query_pipeline.py` |
| 1.4 | Tests: vocabulary expansion, typo correction, inflections, pipeline end-to-end | ✅ Done (50 tests) | `shared/tests/test_vocabulary.py`, `shared/tests/test_query_pipeline.py` |
| 1.5 | Wire pipeline into `execute_search()` with graceful fallback | ✅ Done | `encyclopedia/scripts/encyclopedia.py` |

**Depends on:** Nothing — this phase is self-contained.

### Phase 2: RRF Fusion ✅

**Goal:** Replace priority-based dedup with cross-source-agreement-aware ranking.

| Task | Description | Status | Files |
|------|-------------|--------|-------|
| 2.1 | Implement `RRFEngine` (generic, caller-provided key function) | ✅ Done | `shared/fusion.py` |
| 2.2 | Define per-query-type source weight presets | ✅ Done | `shared/fusion.py:ENCYCLOPEDIA_WEIGHTS` |
| 2.3 | Implement `fuse_sources()` convenience API | ✅ Done | `shared/fusion.py` |
| 2.4 | Tests: single source, cross-source boost, weighted, normalization, filtering | ✅ Done (21 tests) | `shared/tests/test_fusion.py` |
| 2.5 | Wire fusion into `execute_search()` with fallback to priority dedup | ✅ Done | `encyclopedia/scripts/encyclopedia.py` |

**Depends on:** Phase 1 (query type drives weight selection).

### Phase 3: Source Health Monitor

**Goal:** Circuit-break failing sources, probe for recovery, report degradation.

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Implement `SourceHealth` circuit breaker class | `encyclopedia/scripts/source_health.py` (new) |
| 3.2 | Implement `HealthRegistry` managing per-source state | `encyclopedia/scripts/source_health.py` |
| 3.3 | Wire circuit breakers into `execute_search()` dispatch loop | `encyclopedia/scripts/encyclopedia.py` |
| 3.4 | Emit `SOURCE_DEGRADED` / `SOURCE_RESTORED` events | `encyclopedia/scripts/source_health.py` |
| 3.5 | Add cross-type fallback routing when all primary sources are circuit-broken | `encyclopedia/scripts/encyclopedia.py` |
| 3.6 | Tests: circuit break threshold, cooldown, probe recovery, state machine | `encyclopedia/tests/test_source_health.py` (new) |

**Depends on:** Phase 2 (circuit breakers wrap the source dispatch layer).

### Phase 4: Semantic Cache

**Goal:** Avoid redundant backend queries for paraphrased/repeated questions.

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Implement `SemanticCache` with embedding similarity lookup | `encyclopedia/scripts/semantic_cache.py` (new) |
| 4.2 | Implement LRU eviction with bounded entry count (500) | `encyclopedia/scripts/semantic_cache.py` |
| 4.3 | Implement persistence to `~/.encyclopedia/cache/` on shutdown | `encyclopedia/scripts/semantic_cache.py` |
| 4.4 | Wire cache check before dispatch, cache store after fusion | `encyclopedia/scripts/encyclopedia.py` |
| 4.5 | Emit `CACHE_HIT` / `CACHE_MISS` events | `encyclopedia/scripts/semantic_cache.py` |
| 4.6 | Tests: similarity threshold, TTL expiry, LRU eviction, query-type isolation | `encyclopedia/tests/test_semantic_cache.py` (new) |

**Depends on:** `shared.embeddings` (encode function), Phase 2 (caches fused results).

### Phase 5: Source Quality Profiling

**Goal:** Track per-source performance and enable adaptive weight adjustment.

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Implement `SourceProfiler` with rolling window metrics | `encyclopedia/scripts/source_profiler.py` (new) |
| 5.2 | Record result count, latency, fusion rank per query | `encyclopedia/scripts/source_profiler.py` |
| 5.3 | Integrate feedback signals from `shared.feedback.FeedbackCollector` | `encyclopedia/scripts/source_profiler.py` |
| 5.4 | Implement weight adjustment calculator (±5% per window, ±20% cap) | `encyclopedia/scripts/source_profiler.py` |
| 5.5 | Add `ENCYCLOPEDIA_ADAPTIVE_WEIGHTS` feature flag | `shared/feature_flags.py` |
| 5.6 | Wire adaptive weights into RRF fusion call | `encyclopedia/scripts/encyclopedia.py` |
| 5.7 | Tests: profiling accuracy, weight stability, cap enforcement, cold start | `encyclopedia/tests/test_source_profiler.py` (new) |

**Depends on:** Phase 2 (profiles fusion results), `shared.feedback` (existing).

### Phase 6: Integration & Polish

**Goal:** Solidify cross-skill interfaces, update documentation, add observability.

| Task | Description | Files |
|------|-------------|-------|
| 6.1 | Update SKILL.md with new architecture, query syntax, and diagnostics | `encyclopedia/SKILL.md` |
| 6.2 | Extract Section 4 into canonical `constitution.md` | `encyclopedia/constitution.md` (new) |
| 6.3 | Add `--verbose` flag showing query analysis + fusion breakdown | `encyclopedia/scripts/encyclopedia.py` |
| 6.4 | Add `--dry-run` flag showing preprocessing + routing without executing | `encyclopedia/scripts/encyclopedia.py` |
| 6.5 | Verify Rhetoric synergy interface (`rhetoric_request_context`) works end-to-end | `shared/synergies.py`, integration test |
| 6.6 | Verify Volition membrane interface (`encyclopedia.cache.miss`) works end-to-end | `shared/membrane.py`, integration test |
| 6.7 | End-to-end integration tests: full pipeline with mocked backends | `encyclopedia/tests/test_integration.py` (new) |

**Depends on:** All prior phases.

---

## 6. Open Questions

### Q1: Vocabulary Coverage vs. Vocabulary Noise

The dev/programming vocabulary contains ~80 multi-way synonym groups. Adding
more groups improves recall for rare abbreviations but risks expanding common
terms in unwanted ways. For example, should "go" expand to "golang"? In most
programming contexts, yes. In "go to the settings page", absolutely not.

**Current position:** Conservative expansion. Only add synonym groups where the
abbreviation is unambiguous in a dev/programming context. "go" does NOT expand
to "golang" because "go" is a common English word. "k8s" expands to "kubernetes"
because "k8s" has no other plausible meaning. Context-dependent expansion
(using surrounding terms to disambiguate) is a Phase 7+ concern.

### Q2: Cache Similarity Threshold

The threshold of 0.92 for semantic cache hits is a single number governing a
precision/recall tradeoff. Too low: stale results for genuinely different
queries. Too high: cache misses for obvious paraphrases.

**Current position:** 0.92 is a conservative starting point. After Phase 4
ships, measure cache hit rate and false-hit rate (queries that got cached
results but the user immediately re-queried with a different phrasing). Adjust
threshold based on the empirical tradeoff curve.

### Q3: Source Weight Adaptation Stability

Adaptive weights (Phase 5) create a feedback loop: source weights affect which
results rank highly → highly-ranked results get positive feedback → positive
feedback increases source weight. This can create a winner-takes-all dynamic
where one source dominates through positive feedback cycles.

**Current position:** Three safeguards: (1) ±5% adjustment per window prevents
rapid swings, (2) ±20% cap from static baseline prevents runaway drift,
(3) the feedback window is per-query-type, so a source that's great for
`library_docs` doesn't automatically gain weight for `general_search`. If
empirical testing shows instability, reduce the adjustment rate or increase
the window size before disabling the feature.

### Q4: Circuit Breaker vs. Timeout Interaction

A source might be "available" (responds within timeout) but "slow" (consistently
near-timeout). The circuit breaker only triggers on failures, not on slowness.
Should latency spikes also trigger circuit-breaking?

**Current position:** No. Slowness is captured by the quality profiler
(Phase 5), which reduces the source's weight over time. Circuit-breaking is
for hard failures — connection refused, 5xx errors, timeouts. Conflating
slowness with failure leads to circuit-breaking healthy-but-distant backends,
which reduces availability without improving quality.

### Q5: Cross-Reference Depth for Rhetoric

When Rhetoric asks Encyclopedia to cross-reference a claim, how deeply should
Encyclopedia search? A shallow search (limit=3) is fast but might miss the
one contradicting source. A deep search (limit=20) is thorough but adds
latency to Rhetoric's validation pass.

**Current position:** Default to limit=3 with `query_type=library_docs`. If
Rhetoric's Pass 3 (Cross-Reference Integrity) produces too many
`FLAG_INCOMPLETE_REBUTTAL` false negatives (i.e., contradicting evidence
exists but wasn't in the top 3), increase the limit. The synergy interface
accepts an optional `limit` parameter so Rhetoric can request deeper searches
for high-stakes arguments.

### Q6: Multi-Vocabulary Activation

The vocabulary system supports loading different JSON vocabularies for different
domains. But Encyclopedia currently always loads `dev_programming.json`. When
should a different vocabulary activate?

**Current position:** Vocabulary selection is tied to the skill context, not the
query. When Encyclopedia is used within the Cognitive Construct (dev tool),
`dev_programming.json` is the right default. When Encyclopedia is embedded in
Massif Network, `film_production.json` (to be created) would be the default.
This is a deployment-time configuration choice, not a runtime decision.

---

## 7. Success Criteria

### 7.1 Result Quality

Given a test set of 30 queries with human-labeled relevant results:

- **RRF fusion** produces a higher NDCG@5 than priority-based dedup
  on the same query set (the minimum bar — RRF must be strictly better
  than what it replaces)
- **Query preprocessing** produces measurably different (and better) results
  for queries containing abbreviations or misspellings vs. raw dispatch
- **Cross-source agreement** in the top-3 results increases by ≥ 15%
  compared to priority dedup (more sources confirming the top results)

### 7.2 Degradation Handling

- **Zero silent degradation**: every unavailable source appears in
  `degradation.missing[]` or `degradation.errors[]`
- **Circuit breaker recovery**: after a transient failure, sources return
  to healthy status within `cooldown + 1 query` (no permanent damage)
- **Fallback coverage**: when pipeline/fusion imports fail, search still
  returns results using the old code path

### 7.3 Latency

| Component | Target | Measurement |
|-----------|--------|-------------|
| Query preprocessing | < 5ms | `time.perf_counter()` around `pipeline.process()` |
| RRF fusion | < 10ms | `time.perf_counter()` around `engine.fuse_sources()` |
| Semantic cache lookup | < 15ms | `time.perf_counter()` around `check_semantic_cache()` |
| Total pipeline overhead | < 30ms | End-to-end before source dispatch |
| Cache hit path | < 20ms | Query to response when cache hits |

All targets are negligible compared to backend network I/O (200–2000ms
per source). The pipeline must never become the bottleneck.

### 7.4 Safety Invariants

- **Zero fabricated results**: if no source returns data, the response is
  `NOT_FOUND`, never synthesized content
- **Zero suppressed contradictions**: Rhetoric cross-reference always returns
  all topically relevant results, regardless of polarity
- **Zero cross-type cache contamination**: a `library_docs` cache entry is
  never returned for a `general_search` query

### 7.5 Learning Loop (Phase 5+)

After 500 queries with quality profiling:
- Source weight adjustments converge (no oscillation between windows)
- Historically unreliable sources have measurably lower weights than
  reliable ones
- Forced reset of weights to static baseline produces measurably worse
  NDCG@5 than adapted weights (the adaptation is actually helping)

---

## 8. Implementation Notes

### 8.1 Language and Dependencies

- **Python >= 3.12** (consistent with all Cognitive Construct skills)
- **model2vec** (`minishlab/potion-base-8M`): Static embeddings for semantic
  cache and dedup (already in `shared.embeddings`, ~8MB)
- **pydantic**: Data validation (already in `shared.validators`)
- **No new external dependencies**: all infrastructure exists in `shared/`

The vocabulary, pipeline, and fusion modules are pure computation — no I/O,
no side effects, deterministic output. This makes them trivially testable and
safe to call from any async context.

### 8.2 Testing Strategy

**Unit tests** (per-module, ✅ for Phases 1–2):
- `test_vocabulary.py`: 32 tests — expansion, correction, inflection, loading
- `test_query_pipeline.py`: 19 tests — normalization, extraction, classification, end-to-end
- `test_fusion.py`: 21 tests — RRF scoring, cross-source boost, weights, filtering

**Unit tests** (per-module, Phases 3–5):
- `test_source_health.py`: circuit breaker state machine, threshold, cooldown, probe
- `test_semantic_cache.py`: similarity matching, TTL, LRU eviction, type isolation
- `test_source_profiler.py`: metric rolling, weight calculation, cap enforcement

**Integration tests** (Phase 6):
- Full pipeline with mocked backends: preprocessing → dispatch → fusion → response
- Rhetoric synergy: cross-reference request → search → response
- Volition membrane: cache miss event → fallback search → result delivery
- Degradation: simulate source failures → verify circuit breaker → verify recovery

**Property tests** (if time permits):
- For any valid `Vocabulary`, expansion is idempotent: `expand(expand(x)) == expand(x)`
- For any set of source results, RRF fusion is deterministic: same inputs → same scores
- The semantic cache never returns a result for a different `query_type`

### 8.3 File Organization

```
encyclopedia/
  SPEC.md               ← this document
  SKILL.md              ← user-facing skill specification (updated Phase 6)
  constitution.md       ← inviolable rules (created Phase 6)
  resources/
    source_config.json  ← source priorities, routing, timeouts
  scripts/
    encyclopedia.py     ← CLI entrypoint (modified Phases 1–2, further in 3–6)
    source_health.py    ← circuit breaker and health registry (new, Phase 3)
    semantic_cache.py   ← embedding-based query cache (new, Phase 4)
    source_profiler.py  ← rolling quality metrics and weight adaptation (new, Phase 5)
    context7client/     ← existing backend adapter
    exaclient/          ← existing backend adapter
    perplexity/         ← existing backend adapter
  tests/
    test_source_health.py   ← Phase 3 tests
    test_semantic_cache.py  ← Phase 4 tests
    test_source_profiler.py ← Phase 5 tests
    test_integration.py     ← Phase 6 tests
shared/
  vocabulary.py             ← domain vocabulary engine (Phase 1) ✅
  vocabularies/
    dev_programming.json    ← dev/programming terms (Phase 1) ✅
  query_pipeline.py         ← multi-stage preprocessor (Phase 1) ✅
  fusion.py                 ← RRF engine (Phase 2) ✅
  tests/
    test_vocabulary.py      ← 32 tests ✅
    test_query_pipeline.py  ← 19 tests ✅
    test_fusion.py          ← 21 tests ✅
    conftest.py             ← isolated module loading ✅
```

### 8.4 Migration Path

The pipeline and fusion integration is already deployed with import-guarded
fallback:

```python
try:
    from shared.query_pipeline import QueryPipeline, ProcessedQuery
    from shared.fusion import RRFEngine, RRFConfig, ...
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False
```

When `_HAS_PIPELINE` is `False`, `execute_search()` falls back to the original
`classify_query()` → raw dispatch → `compress_for_context()` path. This means:

- **Phase 1–2 changes are already live** and can be observed immediately
- **Phase 3–5 each add a new module** that wraps around existing functionality
  without modifying the core dispatch loop
- **No big-bang cutover** — each phase degrades gracefully when its module is
  unavailable

The only breaking change is that `compress_for_context()` is no longer the
default result ranking when fusion is available. Consumers that depended on
priority-ordered results will see RRF-ordered results instead. Since the
response schema is unchanged (same `SearchResult` structure), this is a
quality improvement, not an API break.

---

## 9. Relationship to Other Skills

Encyclopedia is the knowledge substrate. Its relationship to each skill
is specific and deliberate:

| Skill | Relationship | Interface |
|-------|-------------|-----------|
| **Rhetoric** | Supplies evidence for argument construction; provides contradiction detection for cross-reference integrity (Pass 3) | `rhetoric_request_context()` via synergy bus |
| **Volition** | Provides fallback knowledge when action handlers lack context; receives `cache.miss` events to trigger searches | Membrane absorption + synergy response |
| **Sparring Partner** | No direct interface. Sparring Partner consumes Encyclopedia indirectly through Rhetoric's validated arguments | None |
| **Inland Empire** | Absorbs `search.completed` events for memory storage. Past searches inform future context | Event bus → Inland Empire membrane |
| **Inner-Monologue** | No direct interface. Inner-Monologue may use Encyclopedia results passed through the orchestrating agent | None |

Encyclopedia is the only skill that every other skill can query. It is
deliberately stateless between queries (no conversation memory — that's Inland
Empire's job) and deliberately opinion-free (no argument validation — that's
Rhetoric's job). It finds information. What the agent does with that
information is everyone else's problem.

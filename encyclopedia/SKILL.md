---
name: encyclopedia
description: Knowledge retrieval from multiple sources. Search docs, web, and code with intelligent routing, RRF fusion, circuit breakers, and semantic caching.
license: MIT
metadata:
  version: 3.0.0
  dependencies: python>=3.10
---

# Encyclopedia: The World's Knowledge

> "A billion pages, a billion facts. The Encyclopedia knows all."

## Overview

**Encyclopedia** is the knowledge skill of the Cognitive Construct. It aggregates multiple information sources into a unified, reliable interface: library documentation via Context7, web search via Exa, Perplexity, and (optionally) Kagi, code analysis via repository inspection, and optional advanced sources like SearXNG and CodeGraphContext.

The retrieval pipeline processes every query through six phases:

1. **Query Preprocessing** — spelling correction, synonym expansion, per-backend query adaptation
2. **RRF Fusion** — Reciprocal Rank Fusion treats cross-source agreement as signal, not noise
3. **Source Health** — circuit breakers track per-source failures with automatic recovery probing
4. **Semantic Cache** — embedding similarity avoids redundant queries for near-duplicate inputs
5. **Source Profiling** — rolling window metrics enable adaptive weight adjustment per source
6. **Integration** — verbose diagnostics, dry-run mode, and inter-skill synergies

Each phase degrades gracefully via import guards — if a dependency is missing, the phase is skipped and the pipeline continues with reduced capability.

## Commands

### `search "<query>"`
Perform a knowledge search across all available sources. Include `repo:owner/name` anywhere in the query to automatically pull repository context from `mcp-git-ingest` (and CodeGraphContext when enabled).

```bash
python3 scripts/encyclopedia.py search "repo:anthropics/anthology describe auth middleware"
python3 scripts/encyclopedia.py search "kubernetes auth middleware" --verbose
python3 scripts/encyclopedia.py search "k8s auth" --dry-run
```

**Options:**
- `--sources <list>`: Comma-separated list of sources to query (context7, exa, perplexity, kagi, searxng, codegraph, mcp_git_ingest)
- `--limit <n>`: Maximum results to return (default: 5)
- `--verbose`: Show query analysis, source routing, fusion breakdown, cache status, and health diagnostics
- `--dry-run`: Show preprocessing and routing decisions without executing queries

**Output:**
```json
{
  "status": "success",
  "results": [...],
  "sources_used": ["context7", "exa", "mcp_git_ingest"],
  "degraded": false,
  "degradation": {"missing": [], "errors": []},
  "cached": false
}
```

When `--verbose` is enabled, the response includes additional diagnostics:

```json
{
  "verbose": {
    "query_analysis": {
      "raw_query": "k8s auth middleware",
      "classified_type": "library_docs",
      "corrections": ["k8s → kubernetes"],
      "expansions": [["auth", "authentication", "authorization"]]
    },
    "source_routing": {
      "target_sources": ["context7", "exa"],
      "health_filtered": [],
      "fallback_used": false
    },
    "fusion": {
      "weights": {"context7": 0.52, "exa": 0.48},
      "per_source_results": {"context7": 3, "exa": 4}
    },
    "cache": {"status": "miss"},
    "profiler": {"adaptive_enabled": false}
  }
}
```

When `--dry-run` is enabled, the pipeline stops after preprocessing and routing — no queries are dispatched:

```json
{
  "status": "dry_run",
  "query_analysis": {...},
  "target_sources": ["context7", "exa"],
  "source_health": {"context7": "healthy", "exa": "healthy"},
  "cache_status": "miss"
}
```

### `lookup "<topic>"`
Look up documentation for a specific library, API, or topic.

```bash
python3 scripts/encyclopedia.py lookup "fastapi" --version latest
```

**Options:**
- `--version <ver>`: Specific version to look up (default: latest)

**Output:**
```json
{"status": "success", "topic": "fastapi", "version": "0.115.0", "content": "..."}
```

### `code "<repo_path>" "<query>"`
Analyze a code repository and answer questions about it.

```bash
python3 scripts/encyclopedia.py code "github.com/owner/repo" "how does authentication work"
```

**Options:**
- `--depth <shallow|deep>`: Analysis depth (default: shallow)

**Output:**
```json
{"status": "success", "repository": "owner/repo", "analysis": "..."}
```

## Architecture

### Query Preprocessing (Phase 1)

Raw queries are transformed before dispatch:

1. **Spelling correction** via domain vocabulary (e.g., "kubernets" → "kubernetes")
2. **Synonym expansion** for keyword backends (e.g., "auth" → ["auth", "authentication", "authorization"])
3. **Per-backend adaptation** — Context7 gets library-scoped queries, Exa gets keyword-rich queries, Perplexity gets natural-language questions

Synonym expansion is additive: it never removes or replaces the original query terms (Constitution Rule 7).

### RRF Fusion (Phase 2)

When multiple sources return results, Reciprocal Rank Fusion merges them:

1. **Parallel query** with 5-second timeout per source
2. **RRF scoring** — cross-source agreement boosts results (unlike priority dedup which destroyed agreement signal)
3. **Source weights** — per-query-type weights from `shared.fusion.get_source_weights()`
4. **Metadata** — `fused_from` lists all contributing sources for each result (Constitution Rule 5)

### Source Health Monitor (Phase 3)

Circuit breakers track per-source reliability:

- **3 consecutive failures** → circuit opens (source skipped)
- **60-second cooldown** → probe mode (one test query)
- **Probe success** → healthy; **Probe failure** → circuit re-opens
- Events emitted: `ENCYCLOPEDIA_SOURCE_DEGRADED`, `ENCYCLOPEDIA_SOURCE_RESTORED`

When all primary sources for a query type are circuit-broken, **cross-type fallback routing** attempts sources from adjacent types (e.g., `library_docs` falls back to general search sources).

### Semantic Cache (Phase 4)

Near-duplicate queries hit the cache instead of re-querying backends:

- **Similarity threshold:** 0.92 (configurable via `ENCYCLOPEDIA_CACHE_THRESHOLD`)
- **TTL:** 1 hour (configurable via `ENCYCLOPEDIA_CACHE_TTL`)
- **Max entries:** 500 with LRU eviction
- **Query-type isolation:** a `library_docs` cache entry is never returned for a `general_search` query (Constitution Rule 4)
- **Persistence:** entries saved as JSONL to `~/.encyclopedia/cache/semantic_cache.jsonl`

### Source Quality Profiling (Phase 5)

Rolling window metrics (last 200 queries per source) track performance:

- **Signals:** result count, latency, fusion rank, timeout rate, empty rate, feedback score
- **Weight adjustment:** ±5% per profiling window, capped at ±20% total drift
- **Cold start:** static weights until 50+ samples per source
- **Feature flag:** `ENCYCLOPEDIA_ADAPTIVE_WEIGHTS` (default: disabled)

When enabled, the profiler adjusts fusion weights based on measured performance. When disabled, metrics are still recorded for observability but weights remain static.

## Source Routing

Encyclopedia classifies queries and routes to appropriate sources:

| Query Type | Primary Sources | Fallback Sources |
|------------|-----------------|------------------|
| `library_docs` | context7 | exa, perplexity |
| `general_search` | exa, perplexity | kagi (flagged), searxng |
| `code_context` | mcp-git-ingest (requires `repo:` hint) | exa, context7, CodeGraphContext (optional) |
| `repository` | mcp-git-ingest | exa |

**Classification Strategy:**
1. Explicit type hint: `"doc: React"` or `"code: auth.py"` override routing.
2. Repository hints: `repo:owner/name` automatically trigger `code_context`.
3. Pattern matching: URLs → general_search; `def/class/function` → code_context.
4. Keyword triggers: `"latest"`, `"current"`, `"2024"` → general_search.
5. Default: library_docs.

## Configuration

### Required Environment Variables

Set in `.env.local`:

```bash
# Required (at least one)
EXA_API_KEY=...           # Exa web search
PERPLEXITY_API_KEY=...    # Perplexity AI search

# Optional (enhanced capabilities)
CONTEXT7_API_KEY=...      # Higher rate limits for library docs
KAGI_API_KEY=...          # Kagi search (closed beta)
SEARXNG_URL=...           # Self-hosted SearXNG instance
CGCLI_DB_URL=...          # cgcli SurrealDB (default: surrealkv://~/.local/share/cgcli/codegraph)
```

### Feature Flags

```bash
# Search providers (via ENCYCLOPEDIA_ENABLE_<NAME>=1)
ENCYCLOPEDIA_ENABLE_CONTEXT7=1     # Context7 library docs (default: on)
ENCYCLOPEDIA_ENABLE_KAGI=0         # Kagi search (default: off)
ENCYCLOPEDIA_ENABLE_SEARXNG=0      # SearXNG (default: off)
ENCYCLOPEDIA_ENABLE_CODEGRAPH=0    # SurrealDB code graph (default: off)

# Adaptive features
ENCYCLOPEDIA_ADAPTIVE_WEIGHTS=0    # Source quality profiling (default: off)

# Cache tuning (direct env vars, not feature flags)
ENCYCLOPEDIA_CACHE_THRESHOLD=0.92  # Cosine similarity threshold for cache hits
ENCYCLOPEDIA_CACHE_TTL=3600        # Cache entry TTL in seconds
```

### CLI Tool Paths

Encyclopedia invokes CLI tools for each backend. Override paths via environment:

```bash
CONTEXT7_CLI=context7         # context7 resolve/docs commands
EXA_CLI=exa-mcp-server        # preferred; avoids collision with system `exa` (the `ls` replacement)
KAGI_CLI=kagi                 # kagi search/summarize commands
PERPLEXITY_CLI=perplexity     # perplexity query command
SEARXNG_CLI=searxng           # searxng search command
GIT_INGEST_CLI=mcp-git-ingest # mcp-git-ingest tree/read commands
CGC_CLI=cgc                   # cgc find/analyze commands
```

Note: if `exa` on your machine is the `ls` replacement (often at `/usr/bin/exa`), install `exa-mcp-server` and/or set `EXA_CLI=exa-mcp-server`.

### Credential Validation

Encyclopedia validates credentials at startup and returns clear errors:
- Missing: `"EXA_API_KEY not found"`
- CLI not found: `"cli_not_found"`

## Graceful Degradation

The degradation hierarchy (Constitution Rule 6):

```text
preprocessed query + RRF fusion
  → preprocessed query + priority dedup
    → raw query + priority dedup
      → raw query + single source
        → error
```

Each fallback layer loses quality but preserves availability.

### Degradation Metadata

When any provider is unavailable, the CLI reports:

- `degraded (bool)`: `true` if the request fell back to reduced capability
- `degradation.missing`: structured entries `{source, reason, optional}` describing skipped providers
- `degradation.errors`: runtime failures (timeouts, HTTP errors) with optionality indicators

This metadata makes it clear when optional backends were unavailable and why.

## Error Handling

Encyclopedia returns structured errors without exposing internal details:

```json
{"status": "error", "code": 2, "message": "No results found for query"}
```

Error codes:
- `1`: Configuration error (missing credentials, CLI not found)
- `2`: No results / resource not found
- `3`: Backend unavailable (will use fallback)
- `4`: Internal error

## Synergies

Encyclopedia integrates with other Cognitive Construct skills via the synergy bus:

- **→ Inland Empire**: Frequently accessed topics are cached as memories (`ENCYCLOPEDIA_CACHE_SYNERGY`)
- **← Rhetoric**: Deliberation requests documentation context via `rhetoric_request_context()` (`RHETORIC_ENCYCLOPEDIA_SYNERGY`)

Synergies are enabled by default and controlled by feature flags in `shared/feature_flags.py`.

## Constitution

Encyclopedia's 7 inviolable rules are defined in `encyclopedia/constitution.md`:

1. NEVER fabricate search results
2. NEVER suppress contradicting evidence
3. NEVER silently degrade quality
4. NEVER cache across query types
5. ALWAYS attribute sources
6. ALWAYS prefer graceful degradation over failure
7. NEVER let vocabulary expansion alter meaning

## Files

- `scripts/encyclopedia.py`: Main orchestrator (query pipeline, fusion, dispatch)
- `scripts/source_health.py`: Circuit breaker and health registry
- `scripts/semantic_cache.py`: Embedding similarity cache with LRU eviction
- `scripts/source_profiler.py`: Rolling window metrics and adaptive weights
- `constitution.md`: Inviolable rules governing the knowledge substrate
- `SPEC.md`: Technical specification (design rationale and phase details)
- `~/.encyclopedia/cache/semantic_cache.jsonl`: Persisted cache entries
- `~/.encyclopedia/sessions/`: Query session history

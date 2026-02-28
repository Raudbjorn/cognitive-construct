---
name: inland-empire
description: >-
  Subconscious memory layer. Absorbs observations, surfaces associative memories
  as hypotheses, and builds context across sessions. Commands: remember, consult,
  surface, forget, stats.
license: MIT
metadata:
  version: 2.1.0
  dependencies:
    required:
      - memory_libsql
    optional:
      - openmemory
  env_vars:
    required: []
    optional:
      - LIBSQL_URL
      - LIBSQL_AUTH_TOKEN
      - MEM0_API_KEY
      - POSTGRES_URL
      - INCEPTION_API_KEY
      - INLAND_EMPIRE_STATE_DIR
---

# Inland Empire

> "This is your gut feeling. The raw data of the soul. When logic fails, consult the Empire."

## What It Is

Inland Empire is the **subconscious memory** of the Cognitive Construct. It does not merely store and retrieve — it *absorbs* observations, *associates* related memories, and *surfaces* relevant context without being asked.

Where other memory tools are filing cabinets, Inland Empire is a gut feeling that interrupts you mid-thought: *"This reminds me of something..."*

Inland Empire speaks in **two registers**. When you `consult`, it behaves like a colleague's research — returning evidence you evaluate on its merits. When you `surface`, it behaves like a colleague's hunch — an associative leap that may not be literally true but points toward something your conscious reasoning missed.

This distinction matters for how you *receive* output. Consult results are data to verify. Surface associations are hypotheses to hold lightly — provocations that should inflect your thinking even when they don't directly answer your question.

## How It Works

You interact through five commands. Inland Empire handles storage, routing, and retrieval internally — you never specify backends or worry about where memories live.

### `remember "<text>"`

Commit something to memory. Inland Empire classifies it automatically and stores it in the appropriate backend.

```bash
python3 scripts/inland_empire.py remember "User prefers verbose error messages"
python3 scripts/inland_empire.py remember "The auth flow has a race condition on concurrent logins"
python3 scripts/inland_empire.py remember "Currently debugging the payment webhook timeout"
```

Inland Empire infers the memory type from content:

| Inferred Type | What It Looks Like | Storage |
|---|---|---|
| **fact** | Stable knowledge, preferences, decisions | Structured graph (LibSQL) |
| **pattern** | Recurring observations, behaviors, tendencies | Semantic memory (Mem0) |
| **context** | Session-specific notes, transient state | Local session file (JSONL) |

The `--type` flag exists as an override but should rarely be needed:

```bash
python3 scripts/inland_empire.py remember "Flaky test in CI every Monday" --type pattern
```

**Response:**
```json
{
  "status": "ok",
  "command": "remember",
  "result": {
    "stored": true,
    "inferred_type": "fact"
  }
}
```

### `consult "<query>"`

Actively search memory. Queries all available backends and returns results ranked by relevance.

```bash
python3 scripts/inland_empire.py consult "authentication"
python3 scripts/inland_empire.py consult "user preferences" --depth deep
```

**Options:**
- `--depth shallow|deep`: Controls result count per backend (shallow: 5, deep: 20)
- `--depth shallow|deep`: Controls per-backend result count (shallow: up to 5 results per backend, deep: up to 20 results per backend). The merged results list may contain more than 5 or 20 total items when multiple backends are enabled.

**Response:**
```json
{
  "status": "ok",
  "command": "consult",
  "result": {
    "query": "authentication",
    "depth": "shallow",
    "query": "user preferences",
    "depth": "deep",
      {
        "summary": "The auth flow has a race condition on concurrent logins",
        "type": "pattern",
        "score": 0.87,
        "observed_at": "2025-12-10T14:30:00Z"
      },
      {
        "summary": "Auth service runs on port 8080 behind nginx proxy",
        "type": "fact",
        "score": null,
        "observed_at": null
      }
    ],
    "partial": false
  }
}
```

The `partial` flag indicates whether any backend timed out or was unavailable. When `true`, results may be incomplete.

Consult results are **evidence** — evaluate them the way you'd evaluate search results. Check relevance, check recency, use or discard.

### `surface "<context>"`

Ask Inland Empire what it associates with the current situation. Unlike `consult` (which searches for specific terms), `surface` casts a wide net across all memory types looking for anything tangentially relevant.

This is the "gut feeling" command — use it when starting a new task, returning from a break, or when something feels familiar but you can't place it.

```bash
python3 scripts/inland_empire.py surface "refactoring the payment module"
```

#### Interpreting Associations

Surface results are **hypotheses**, not evidence. They are associative leaps that may be metaphorical, tangential, or wrong — but they point toward something your linear reasoning may have missed.

1. **Look for the hidden thread.** Three associations that seem unrelated often share an underlying pattern. The connection between them is the signal, not any individual result.
2. **Let associations inflect your thinking** even when the topic doesn't obviously match. A surfaced memory about DNS failures while you're debugging payments isn't noise — it's saying *"this has the shape of a resolution problem."*
3. **Hold relevance scores lightly.** A `medium` relevance association may be the most important one. The scoring reflects semantic distance, not diagnostic value.

**Wrong:** *"Inland Empire surfaced 3 memories: [list each one with summary]."*

**Right:** *"The surfaced associations share a pattern — they all involve systems that failed silently at integration boundaries. This payment refactor touches two integration boundaries."*

**Response:**
```json
{
  "status": "ok",
  "command": "surface",
  "result": {
    "context": "refactoring the payment module",
    "associations": [
      {
        "summary": "Payment webhook timeout issue was caused by missing retry logic",
        "type": "pattern",
        "relevance": "high"
      },
      {
        "summary": "User prefers explicit error types over generic exceptions",
        "type": "fact",
        "relevance": "medium"
      }
    ],
    "voice": null,
    "partial": false
  }
}
```

The `voice` field is always present but nullable. When non-null, it carries an orthogonal reading of the associations — a gut-level interpretation that may be metaphorical, impressionistic, or oblique. It is not a summary of the data above. Treat it as one more associative signal: absorb it, don't paraphrase it.

### `forget "<query>"`

Selectively remove memories. Useful for clearing stale context, correcting wrong information, or pruning noise.

```bash
python3 scripts/inland_empire.py forget "payment webhook timeout"
python3 scripts/inland_empire.py forget --type context --before 7d
```

**Options:**
- `--type fact|pattern|context`: Restrict to one memory type
- `--before <duration>`: Forget entries older than duration (e.g., `7d`, `30d`)
- `--dry-run`: Show what would be forgotten without deleting

### `stats`

Backend health and memory statistics.

```bash
python3 scripts/inland_empire.py stats
```

**Response:**
```json
{
  "status": "ok",
  "command": "stats",
  "result": {
    "version": "2.1.0",
    "version": "2.1.0",
      "graph": {"status": "available", "mode": "local"},
      "semantic": {"status": "available", "mode": "hosted"},
      "session": {"status": "available", "entries": 42},
      "voice": {"status": "available", "model": "mercury-small", "provider": "inception"}
    }
  }
}
```

## Storage Architecture

Inland Empire unifies multiple backends behind a single interface. The user never selects a backend — routing is internal.

```text
                  ┌─────────────────────────┐
                  │     Inland Empire        │
                  │   (classify + route)     │
                  └────┬────────┬────────┬───┘
                       │        │        │
                ┌──────▼──┐ ┌───▼────┐ ┌─▼──────────┐
                │  Graph  │ │Semantic│ │  Session    │
                │ (facts) │ │(patterns)│ │ (context)  │
                ├─────────┤ ├────────┤ ├────────────┤
                │ LibSQL  │ │  Mem0  │ │   JSONL    │
                │ SQLite  │ │ Cloud  │ │   local    │
                └─────────┘ └────────┘ └────────────┘
                  always      optional     always
```

**Backend detection** (from environment):
- **Graph**: Always available. Uses `LIBSQL_URL` if set, otherwise local SQLite.
- **Semantic**: Requires `MEM0_API_KEY`. Gracefully disabled when not set — patterns stored in graph as fallback.
- **Session**: Always available. Local JSONL file in `INLAND_EMPIRE_STATE_DIR`.

## When to Use

- **Remember** something that doesn't belong in code or config files
- **Consult** before making decisions that might repeat past mistakes — results are facts to evaluate
- **Surface** for the question *what am I not seeing?* — when starting unfamiliar work, when something feels familiar but you can't place it, or when linear reasoning has stalled
- **Forget** stale context that's no longer relevant
- Build institutional memory that **survives context window compression**

## When NOT to Use

- Structured project data → use files, databases, or config
- Code-level documentation → use comments and docstrings
- Ephemeral debugging notes → use the session scratchpad, not persistent memory
- Anything that needs to be version-controlled → use git

## Synergies

Inland Empire integrates with other Cognitive Construct skills:

- **← Rhetoric**: Significant deliberation outcomes are worth remembering
- **← Encyclopedia**: Research findings can be stored as facts
- **→ Rhetoric**: Surfaced memories inform deliberation context
- **→ Volition**: Recalled patterns guide action selection

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `LIBSQL_URL` | Graph database URL | `file:./memory-tool.db` (local) |
| `LIBSQL_AUTH_TOKEN` | Remote Turso auth | (none) |
| `MEM0_API_KEY` | Mem0 Cloud API key | (none — semantic backend disabled) |
| `POSTGRES_URL` | Self-hosted Mem0 (reserved, not yet supported) | (none) |
| `INCEPTION_API_KEY` | Mercury diffusion LLM key (voice layer) | (none — voice disabled) |
| `INLAND_EMPIRE_STATE_DIR` | Storage directory override | current directory |

## Error Handling

All commands return JSON with `"status": "ok"` or `"status": "error"`:

```json
{
  "status": "error",
  "command": "remember",
  "error": {
    "message": "Not initialized",
    "code": "NOT_INITIALIZED"
  }
}
```

Backend failures are isolated — if one backend fails, others still return results. The `partial` flag in responses indicates incomplete results.

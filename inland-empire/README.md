# Inland Empire

> "This is your gut feeling. The raw data of the soul. When logic fails, consult the Empire."

A subconscious memory layer for the Cognitive Construct. Absorbs observations, surfaces relevant memories, and builds associative context across sessions.

## Quick Start

```bash
# Check backend status
python3 scripts/inland_empire.py stats

# Store memories (type is auto-classified)
python3 scripts/inland_empire.py remember "User prefers verbose errors"
python3 scripts/inland_empire.py remember "Currently debugging the login flow"
python3 scripts/inland_empire.py remember "Auth timeout keeps happening under load"

# Search memories
python3 scripts/inland_empire.py consult "authentication" --depth deep

# Get gut feelings about current work
python3 scripts/inland_empire.py surface "refactoring the payment module"

# Clean up stale context
python3 scripts/inland_empire.py forget --type context --before 7d
```

## Architecture

Inland Empire unifies multiple backends behind a single interface. You never select a backend — routing is internal.

```text
                  ┌─────────────────────────┐
                  │     Inland Empire        │
                  │  (classify + route)      │
                  └────┬────────┬────────┬───┘
                       │        │        │
                ┌──────▼──┐ ┌───▼────┐ ┌─▼──────────┐
                │  Graph  │ │Semantic│ │  Session    │
                │ (facts) │ │(pattern)│ │(context)   │
                ├─────────┤ ├────────┤ ├─────────────┤
                │ LibSQL  │ │  Mem0  │ │    JSONL    │
                │ SQLite  │ │ Cloud  │ │    local    │
                └─────────┘ └────────┘ └─────────────┘
                  always      optional     always
```

**Backend detection** (from environment):
- **Graph**: Always available. Uses `LIBSQL_URL` if set, otherwise local SQLite.
- **Semantic**: Requires `MEM0_API_KEY`. When disabled, patterns fall back to the graph backend.
- **Session**: Always available. Local JSONL file.

## Commands

### `remember "<text>"`

Commit something to memory. Inland Empire classifies the type automatically from content:

| Inferred Type | Signals | Storage |
|---|---|---|
| **fact** | Default — stable knowledge | Graph (LibSQL) |
| **pattern** | "always", "recurring", "prefers", "flaky", etc. | Semantic (Mem0) or graph fallback |
| **context** | "currently", "debugging", "working on", "blocked on", etc. | Session (JSONL) |

```bash
# Auto-classified as fact
python3 scripts/inland_empire.py remember "Auth service runs on port 8080"

# Auto-classified as pattern ("prefers")
python3 scripts/inland_empire.py remember "User prefers tabs over spaces"

# Auto-classified as context ("currently")
python3 scripts/inland_empire.py remember "Currently debugging the payment webhook"

# Override auto-classification
python3 scripts/inland_empire.py remember "Flaky test in CI" --type fact
```

### `consult "<query>"`

Actively search stored memories across all backends.

```bash
python3 scripts/inland_empire.py consult "authentication"
python3 scripts/inland_empire.py consult "user preferences" --depth deep --type pattern
```

**Options:**
- `--depth shallow|deep`: Result count per backend (shallow: 5, deep: 20)
- `--type fact|pattern|context`: Filter to one memory type

### `surface "<context>"`

Broad associative retrieval — the "gut feeling" command. Casts a wide net across all memory types looking for anything tangentially relevant.

```bash
python3 scripts/inland_empire.py surface "refactoring the payment module"
```

### `forget`

Selectively remove memories by query, type, age, or a combination.

```bash
# Delete by query match
python3 scripts/inland_empire.py forget "payment webhook"

# Delete old session context
python3 scripts/inland_empire.py forget --type context --before 7d

# Preview before deleting
python3 scripts/inland_empire.py forget "stale notes" --dry-run
```

**Options:**
- `--type fact|pattern|context`: Restrict to one memory type
- `--before <duration>`: Delete entries older than duration (e.g., `7d`, `24h`, `30m`)
- `--dry-run`: Show what would be deleted without actually deleting

### `stats`

Backend health and memory statistics.

```bash
python3 scripts/inland_empire.py stats
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `LIBSQL_URL` | Graph database URL | `file:./memory-tool.db` (local) |
| `LIBSQL_AUTH_TOKEN` | Remote Turso auth | (none) |
| `MEM0_API_KEY` | Mem0 Cloud API key | (none — semantic disabled) |
| `POSTGRES_URL` | Self-hosted Mem0 (reserved, not yet supported) | (none) |
| `INCEPTION_API_KEY` | Voice layer API key (Mercury diffusion LLM) | (none — voice disabled) |
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

Backend failures are isolated — if one backend fails, others still return results. The `partial` flag in responses indicates incomplete results. When the semantic backend is unavailable, patterns fall back to graph storage automatically.

## Directory Structure

```text
inland-empire/
├── scripts/
│   ├── inland_empire.py      # Main CLI entrypoint
│   ├── memory_libsql/        # LibSQL graph client
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── types.py
│   │   └── result.py
│   ├── memory_graph/          # JSONL graph client (library, not used directly)
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── types.py
│   │   └── result.py
│   └── mem0/                  # Mem0 client
│       └── openmemory/
│           ├── __init__.py
│           ├── client.py
│           ├── types.py
│           └── result.py
├── SKILL.md                   # Skill manifest
├── README.md                  # This file
├── .env.example               # Configuration template
├── memory-tool.db             # LibSQL database (created on first use)
└── session_memory.jsonl       # Session memory (created on first use)
```

## Dependencies

### Required
- `memory_libsql` (included)
- `libsql_experimental` (`uv pip install libsql_experimental`)

### Optional
- `openmemory` (included, for semantic backend)
- `httpx` (for Mem0 API calls)

## License

MIT

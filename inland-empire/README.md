# Inland Empire

> "This is your gut feeling. The raw data of the soul. When logic fails, consult the Empire."

A unified memory substrate for Claude skills. Store facts, patterns, and context across multiple backends with a single CLI interface.

## Quick Start

```bash
# Set environment variables
source /path/to/.env.local

# Check backend status
python3 inland-empire.py stats

# Store memories
python3 inland-empire.py remember "User prefers verbose errors" --type pattern
python3 inland-empire.py remember "Database uses normalized schema" --type fact
python3 inland-empire.py remember "Session note about auth flow" --type context

# Query memories
python3 inland-empire.py consult "user preferences" --depth deep
```

## Architecture

Inland Empire unifies three memory backends behind sanitized aliases:

| Alias | Backend | Storage | Use Case |
|-------|---------|---------|----------|
| `fact_memory` | memory_libsql | LibSQL/SQLite graph | Structured facts, entities, relations |
| `pattern_memory` | mem0/openmemory | Hosted API or Postgres | Learned patterns, preferences |
| `context_memory` | JSONL file | Local session file | Session context, transient notes |

```
┌─────────────────────────────────────────────────────────┐
│                    inland-empire.py                      │
│                    (Unified CLI)                         │
├─────────────────┬─────────────────┬─────────────────────┤
│   fact_memory   │  pattern_memory │   context_memory    │
│  (memory_libsql)│   (openmemory)  │      (JSONL)        │
├─────────────────┼─────────────────┼─────────────────────┤
│  LibSQL/SQLite  │   Mem0 Cloud    │  session_memory.jsonl│
│  Graph Storage  │   or Postgres   │                     │
└─────────────────┴─────────────────┴─────────────────────┘
```

## Commands

### `remember`

Store a memory to a specific backend.

```bash
python3 inland-empire.py remember "<text>" [--type fact|pattern|context]
```

**Options:**
- `--type`, `-t`: Memory type (default: `fact`)
  - `fact`: Structured knowledge (entities/relations)
  - `pattern`: Learned preferences and behaviors
  - `context`: Session-specific notes

**Examples:**
```bash
# Store a fact (default)
python3 inland-empire.py remember "Auth service runs on port 8080"

# Store a pattern
python3 inland-empire.py remember "User prefers tabs over spaces" --type pattern

# Store session context
python3 inland-empire.py remember "Currently debugging login flow" --type context
```

### `consult`

Query stored memories across backends.

```bash
python3 inland-empire.py consult "<query>" [--depth shallow|deep] [--type fact|pattern|context]
```

**Options:**
- `--depth`, `-d`: Search depth (default: `shallow`)
  - `shallow`: Return up to 5 results
  - `deep`: Return up to 20 results
- `--type`, `-t`: Filter by memory type (optional, queries all if omitted)

**Examples:**
```bash
# Search all backends
python3 inland-empire.py consult "authentication"

# Deep search patterns only
python3 inland-empire.py consult "user preferences" --depth deep --type pattern
```

**Response Format:**
```json
{
  "status": "ok",
  "command": "consult",
  "result": {
    "query": "authentication",
    "depth": "shallow",
    "results": [
      {
        "origin": "fact",
        "summary": "Auth service runs on port 8080",
        "score": null,
        "observed_at": null,
        "backend": "fact_memory",
        "partial": false,
        "metadata": { "entity_name": "fact_1234", "entity_type": "fact" }
      }
    ],
    "metadata": {
      "requested_backends": ["fact_memory", "pattern_memory", "context_memory"],
      "completed_backends": ["fact_memory", "pattern_memory", "context_memory"],
      "unavailable_backends": [],
      "timed_out_backends": [],
      "partial": false
    }
  }
}
```

### `stats`

Display backend health and statistics.

```bash
python3 inland-empire.py stats
```

**Response Format:**
```json
{
  "status": "ok",
  "command": "stats",
  "result": {
    "version": "1.0.0",
    "state_dir": "/path/to/inland-empire",
    "backends": {
      "fact_memory": {
        "status": "available",
        "backend": "memory_libsql",
        "url": "file:./memory-tool.db (local)",
        "remote": false
      },
      "pattern_memory": {
        "status": "available",
        "backend": "mem0",
        "mode": "hosted"
      },
      "context_memory": {
        "status": "available",
        "backend": "jsonl",
        "file": "/path/to/session_memory.jsonl",
        "entries": 42
      }
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Backend | Description |
|----------|---------|-------------|
| `LIBSQL_URL` | fact_memory | LibSQL database URL (default: `file:./memory-tool.db`) |
| `LIBSQL_AUTH_TOKEN` | fact_memory | Auth token for remote Turso connections |
| `MEM0_API_KEY` | pattern_memory | Mem0 Cloud API key (enables hosted mode) |
| `POSTGRES_URL` | pattern_memory | Postgres URL (enables self-hosted mode) |
| `INLAND_EMPIRE_STATE_DIR` | all | Override storage directory |

### Backend Detection

- **fact_memory**: Always available (falls back to local SQLite)
- **pattern_memory**:
  - `MEM0_API_KEY` set → hosted mode
  - `POSTGRES_URL` set → self-hosted mode
  - Neither → disabled
- **context_memory**: Always available (local JSONL file)

## Dependencies

### Required
- `memory_libsql` (included)
- `memory_graph` (included)
- `libsql_experimental` (`uv pip install libsql_experimental`)

### Optional
- `openmemory` (included, for pattern_memory)
- `httpx` (for mem0 API calls)

## Directory Structure

```
inland-empire/
├── inland-empire.py      # Main CLI entrypoint
├── SKILL.md              # Skill manifest
├── README.md             # This file
├── memory_libsql/        # LibSQL graph client
│   ├── __init__.py
│   ├── client.py
│   ├── types.py
│   └── result.py
├── memory_graph/         # JSONL graph client
│   ├── __init__.py
│   ├── client.py
│   ├── types.py
│   └── result.py
├── mem0/                 # Mem0 client
│   └── openmemory/
│       ├── __init__.py
│       ├── client.py
│       ├── types.py
│       └── result.py
├── memory-tool.db        # LibSQL database (created on first use)
└── session_memory.jsonl  # Context memory file (created on first use)
```

## Error Handling

All commands return JSON with a `status` field:
- `"ok"`: Operation succeeded
- `"error"`: Operation failed

Error responses include:
```json
{
  "status": "error",
  "command": "remember",
  "error": {
    "message": "Backend not available: memory_libsql dependency missing",
    "code": "BACKEND_UNAVAILABLE",
    "backend": "fact_memory"
  }
}
```

## Use Cases

- **Store hunches and preferences** that don't belong in code files
- **Recall project context** after breaks or context switches
- **Detect patterns** across sessions (via mem0's semantic search)
- **Build institutional memory** for recurring issues and solutions
- **Session notes** for complex debugging or refactoring tasks

## License

MIT

# Cognitive Construct Design

Architecture for a modular AI enhancement system comprising four Claude Skills that provide distinct cognitive capabilities.

## Overview

The Cognitive Construct organizes AI capabilities into four domains:

| Skill | Domain | Purpose |
|-------|--------|---------|
| **Encyclopedia** | Knowledge | Unified search across documentation, web, and code |
| **Rhetoric** | Reasoning | Structured thought, multi-model deliberation, bias detection |
| **Inland Empire** | Memory | Persistent facts, patterns, and context across sessions |
| **Volition** | Agency | Semantic code editing, LLM consultation, web search |

Each skill wraps multiple backends behind an opaque CLI interface, enabling Claude to leverage complex capabilities without managing implementation details.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Claude                               │
│                    (or other LLM)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │ CLI Commands
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cognitive Construct                       │
├────────────────┬────────────────┬────────────────┬──────────┤
│  Encyclopedia  │    Rhetoric    │ Inland Empire  │ Volition │
│   (Knowledge)  │   (Reasoning)  │   (Memory)     │ (Agency) │
├────────────────┼────────────────┼────────────────┼──────────┤
│ search <query> │ think <text>   │ remember <text>│ act <act>│
│ lookup <topic> │ deliberate <q> │ consult <query>│ edit <s> │
│ code <repo>    │ review         │ stats          │ query <s>│
└───────┬────────┴───────┬────────┴───────┬────────┴────┬─────┘
        │                │                │              │
        ▼                ▼                ▼              ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ context7      │ │ ai-counsel    │ │ memory_libsql │ │ serena (LSP)  │
│ exa           │ │ sequential-   │ │ mem0/openem.  │ │ cross-llm-mcp │
│ perplexity    │ │   thinking    │ │ JSONL context │ │ websearch     │
│ git-ingest    │ │ vibe-check    │ │               │ │ shodan        │
│ (kagi, searxng)│ │              │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

## Design Philosophy

### Why Wrapper Skills?

Claude has native MCP support. The natural question: why add a wrapper layer?

**What wrappers provide:**

1. **Intelligent Source Selection** — Claude shouldn't decide between context7/exa/perplexity. The skill routes based on query type.

2. **Result Aggregation** — Multiple sources for the same query, deduplicated and ranked by relevance.

3. **Unified Memory Model** — Three persistence backends (graph, semantic, session) exposed as one `remember/consult` interface.

4. **Credential Isolation** — API keys managed without LLM exposure.

5. **Graceful Degradation** — When one backend fails, others compensate transparently.

**What this is NOT:**
- Opacity for opacity's sake
- Adding abstraction without purpose
- Hiding capabilities Claude already knows exist

### Architectural Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| CLI facade over direct MCP | +1 abstraction layer | Enables routing, merging, fallbacks |
| File-based persistence | Slower than in-memory | Survives sessions, no external DB |
| Multiple backends per skill | Maintenance burden | Redundancy, best-of-breed per query |
| Opaque responses | Less transparency | Simpler LLM interface |

## Skill Details

### Encyclopedia (Knowledge)

Unified knowledge retrieval with intelligent routing.

**Backends:**
- `context7` — Library documentation (when library is indexed)
- `exa` — Web and code search (general fallback)
- `perplexity` — AI-powered search
- `mcp-git-ingest` — Repository structure analysis
- `kagi` — High-quality search (optional, closed beta)
- `searxng` — Meta search (optional, self-hosted)
- `CodeGraphContext` — Code graph analysis (optional, requires Neo4j)

**Commands:**
```bash
encyclopedia.py search "React hooks best practices"
encyclopedia.py search "repo:anthropics/claude-code auth"
encyclopedia.py lookup "fastapi" --version 0.100
encyclopedia.py code "github.com/sveltejs/svelte" "runes"
```

**Query Routing:**

| Query Type | Primary | Fallback |
|------------|---------|----------|
| `library_docs` | context7 | exa |
| `general_search` | exa, perplexity | kagi, searxng |
| `code_context` | git-ingest | exa, codegraph |

**Classification:**
1. Explicit prefixes: `doc:`, `code:`, `web:`
2. Repository hints: `repo:owner/name`
3. Pattern matching: URLs → general; `def/class` → code
4. Keywords: `latest`, `2024` → general
5. Default: library docs

### Rhetoric (Reasoning)

Structured reasoning with thought tracking and multi-model deliberation.

**Backends:**
- `sequential-thinking` — Thought recording with revision tracking
- `ai-counsel` — Multi-model deliberation engine
- `vibe-check` — Pattern detection and bias assessment

**Commands:**
```bash
rhetoric.py think "Factory pattern seems overkill here" --session-id proj-1
rhetoric.py deliberate "Microservices or monolith for MVP?" --rounds 2
rhetoric.py review --session-id proj-1
rhetoric.py status
```

**Deliberation:**

Requires 2+ model API keys. Supported providers:
- OpenAI (gpt-4o, gpt-4o-mini)
- Anthropic (claude-sonnet-4, claude-3)
- OpenRouter (any model)
- Ollama (local models)
- Google (Gemini)

**Features:**
- Thought revision history and branching
- Convergence detection with similarity scoring
- Cognitive bias detection via vibe-check
- Markdown deliberation transcripts

### Inland Empire (Memory)

> "This is your gut feeling. The raw data of the soul."

Unified memory substrate with three backend types.

**Backends:**

| Alias | Backend | Storage | Use Case |
|-------|---------|---------|----------|
| `fact_memory` | memory_libsql | LibSQL/SQLite graph | Entities, relations |
| `pattern_memory` | mem0/openmemory | Cloud API or Postgres | Learned patterns |
| `context_memory` | JSONL file | Local session file | Session notes |

**Commands:**
```bash
inland-empire.py remember "Auth service runs on port 8080" --type fact
inland-empire.py remember "User prefers tabs over spaces" --type pattern
inland-empire.py remember "Currently debugging login" --type context
inland-empire.py consult "authentication" --depth deep
inland-empire.py stats
```

**Backend Detection:**
- `fact_memory`: Always available (local SQLite fallback)
- `pattern_memory`: `MEM0_API_KEY` → hosted; `POSTGRES_URL` → self-hosted
- `context_memory`: Always available (local JSONL)

**Use Cases:**
- Store hunches that don't belong in code files
- Recall project context after breaks
- Detect patterns across sessions
- Build institutional memory for recurring issues

### Volition (Agency)

Executive skill that transforms intent into action.

**Backends:**
- `serena` — LSP-powered semantic code editing
- `cross-llm-mcp` — Multi-provider LLM consultation
- `openai-websearch-mcp` — Reasoning-enhanced web search
- `mcp-shodan` — Security reconnaissance (rate-limited, requires confirmation)

**Commands:**
```bash
volition.py act "refactor authentication for better security"
volition.py edit "UserAuth.validate" "add rate limiting"
volition.py query web "Python 3.13 features"
volition.py query llm "JWT rotation best practices"
volition.py query security "exposed MongoDB" --confirm
volition.py capabilities
```

**Safety Constraints:**
- Security queries require explicit `--confirm` flag
- Rate limiting on Shodan calls
- Audit logging for security operations

## Credential Management

### Required Credentials

```bash
# Encyclopedia (at least one required)
EXA_API_KEY=...
PERPLEXITY_API_KEY=...

# Rhetoric (at least two required for deliberation)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Inland Empire (at least one for pattern_memory)
MEM0_API_KEY=...  # or POSTGRES_URL
```

### Optional Credentials

```bash
# Encyclopedia extras
CONTEXT7_API_KEY=...      # Higher rate limits
KAGI_API_KEY=...          # Closed beta
SEARXNG_URL=...           # Self-hosted
NEO4J_URI=...             # CodeGraph

# Rhetoric extras
OPENROUTER_API_KEY=...
GOOGLE_CLOUD_API_KEY=...
OLLAMA_URL=...

# Volition extras
SHODAN_API_KEY=...
```

### Feature Flags

```bash
ENCYCLOPEDIA_ENABLE_CONTEXT7=true
ENCYCLOPEDIA_ENABLE_KAGI=false
ENCYCLOPEDIA_ENABLE_SEARXNG=false
ENCYCLOPEDIA_ENABLE_CODEGRAPH=false
```

## Response Format

All skills return consistent JSON:

### Success
```json
{
  "status": "success",
  "command": "search",
  "result": {
    "query_type": "library_docs",
    "results": [...],
    "sources_used": ["context7", "exa"],
    "degraded": false
  }
}
```

### Error
```json
{
  "status": "error",
  "command": "search",
  "error": {
    "code": "BACKEND_UNAVAILABLE",
    "message": "Search service temporarily unavailable",
    "backend": "context7"
  }
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error |
| 2 | Resource not found / no results |
| 3 | Backend unavailable |
| 4 | Permission denied / internal error |

## Inter-Skill Synergies

Skills can call each other through their CLIs:

```
Encyclopedia ←→ Rhetoric
     │              │
     └──→ Inland Empire ←──┘
              │
              ↓
          Volition
```

**Example flows:**

1. **Rhetoric → Encyclopedia**: Deliberation fetches context before reasoning
2. **Volition → Inland Empire**: Actions logged for pattern detection
3. **Encyclopedia → Inland Empire**: Search results cached for recall
4. **Rhetoric → Inland Empire**: Thought patterns stored for review

## Future: Emergence Architecture

Current architecture is request/response. For emergence, skills would need:

### Blackboard Pattern

Instead of skill-calls-skill:
- Skills watch a shared environment (Inland Empire as blackboard)
- Skills post contributions when they have something
- No explicit orchestrator—self-organization emerges

```
                    BLACKBOARD
    ┌────────────────────────────────────────┐
    │  [knowledge_gap: React hooks]          │
    │  [thought: Factory pattern overkill]   │
    │  [pattern: User prefers verbose]       │
    │  [action: Refactored auth module]      │
    └────────────────────────────────────────┘
         ↕         ↕          ↕          ↕
    Encyclopedia  Rhetoric   Inland    Volition
```

### Membrane Model

Each skill defines what it absorbs/emits:

```python
encyclopedia_membrane = Membrane(
    absorbs=["knowledge_gap", "query"],
    emits=["knowledge_found", "no_results"]
)

rhetoric_membrane = Membrane(
    absorbs=["decision_needed", "knowledge_found"],
    emits=["thought", "question", "knowledge_gap"]
)
```

**Key insight:** Rhetoric emits `knowledge_gap`, Encyclopedia absorbs it. No direct call—the blackboard mediates.

### Requirements for Emergence

- Long-running processes (not discrete CLI calls)
- Shared mutable state (the blackboard)
- Async everything
- Persistence across sessions
- Feedback signal (what worked vs. what didn't)

This is deferred to future phases. Current architecture validates skill boundaries and MCP integration with working tools.

---

*For implementation details, see individual skill READMEs in `skills/skills/*/README.md`.*

---
name: volition
description: Agency and execution. Edit code semantically, invoke LLMs, search the web, and query security services.
version: 6.0.0
license: MIT
metadata:
  phase: 6
  dependencies: python>=3.12, httpx, pydantic, model2vec
---

# Volition: The Will to Act

> "Between stimulus and response there is a space. In that space is our power
> to choose our response. In our response lies our growth and our freedom."
> — Viktor Frankl

## Overview

**Volition** is the executive function of the Cognitive Construct. It transforms intent into deliberate, recoverable action through a five-stage pipeline: classify intent, build an action plan, validate before acting, execute with fallback recovery, and learn from outcomes.

Volition replaces reactive keyword matching with semantic understanding. When you say "make the auth module more secure," it routes to code editing — not to Shodan — because it understands meaning, not just keywords.

## Architecture

Every action flows through this pipeline:

```text
user prompt → Intent Classifier (embedding + keyword fusion)
                    |
           confidence >= threshold?
              yes /     \ no
               |          |
        Plan Builder    Clarification
        (DAG steps)     Request (top-2)
               |
        Pre-flight Validator (4-pass)
               |
          all passes OK?
            yes /    \ no
             |        flags → user
        Executor (step by step, with fallbacks)
             |
        Outcome Logger (audit + feedback)
             |
        result → user
```

The classifier fuses two signals via Reciprocal Rank Fusion (RRF): embedding similarity (Model2Vec, weighted 0.7) and keyword overlap (weighted 0.3). Historical feedback from prior executions provides a conservative adjustment (+-10%).

## Commands

### `act "<action>"`
Execute a general action. Volition classifies intent, builds an action plan, validates it, and executes.

```bash
python3 volition.py act "refactor the authentication module for better security"
```

**Output:**
```json
{
  "status": "success",
  "handler": "code_edit",
  "confidence": 0.71,
  "plan_id": "plan-a1b2c3d4",
  "outcomes": [
    {"step_id": "step-1", "handler": "code_edit", "status": "success", "summary": "..."}
  ]
}
```

**Options:**
- `--handler <name>`: Override automatic routing (`code_edit`, `llm_call`, `web_search`, `security`)
- `--confirm`: Pre-confirm security actions
- `--verbose`: Show full classification breakdown with embedding, keyword, and fused scores for all candidates
- `--dry-run`: Show the action plan and pre-flight result without executing

### `edit "<symbol>" "<change>"`
Perform semantic code edits using LSP-powered tools. Finds symbols by name and applies changes contextually.

```bash
python3 volition.py edit "UserAuth.validate" "add rate limiting check at start"
```

**Options:**
- `--project <path>`: Project root (default: current directory)
- `--fallback`: Allow text-based editing if LSP unavailable

**Output:**
```json
{"status": "success", "symbol": "UserAuth.validate", "file": "src/auth.py", "changes": 1}
```

### `query "<service>" "<query>"`
Query external services for information.

**Services:**
- `web`: General web search via openai-websearch-mcp
- `security`: Security reconnaissance via Shodan (requires confirmation)
- `llm`: Consult an LLM for expert advice

```bash
# Web search
python3 volition.py query web "latest Python 3.13 features"

# Security query (explicit confirmation required)
python3 volition.py query security "exposed MongoDB instances in AS12345" --confirm

# LLM consultation
python3 volition.py query llm "best practices for JWT token rotation" --tag coding
```

**Options:**
- `--provider <name>`: Override LLM provider for `llm` service
- `--tag <tag>`: Task tag for LLM routing (coding, reasoning, creative, general)
- `--confirm`: Pre-confirm security queries (use with caution)

### `capabilities`
List available capabilities and their status.

```bash
python3 volition.py capabilities
```

**Output:**
```json
{
  "code_editing": {"status": "available", "backend": "serena", "languages": ["python", "typescript", "rust"]},
  "llm_consultation": {"status": "available", "providers": ["openai", "anthropic", "deepseek"]},
  "web_search": {"status": "available", "backend": "openai-websearch"},
  "security_queries": {"status": "restricted", "backend": "shodan", "rate_limit": "10/hour"}
}
```

## Confidence Gating

When the classifier is not confident enough to route (fused score below threshold, default 0.65), Volition asks instead of guessing:

```json
{
  "status": "clarification_required",
  "message": "Intent is ambiguous. Did you mean one of these?",
  "candidates": [
    {"handler": "code_edit", "confidence": 0.52, "embedding_score": 0.55, "keyword_score": 0.49},
    {"handler": "security", "confidence": 0.48, "embedding_score": 0.46, "keyword_score": 0.51}
  ],
  "action": "your original prompt text"
}
```

This is Constitution Rule 1: uncertainty is information, not a bug. Override with `--handler` if you know what you want.

## Multi-Step Plans

Compound actions ("review the auth module and fix any issues") are split into a DAG of steps with dependency ordering:

```json
{
  "plan_id": "plan-a1b2c3d4",
  "steps": [
    {"step_id": "step-1", "handler": "llm_call", "action": "Review the authentication module", "depends_on": []},
    {"step_id": "step-2", "handler": "code_edit", "action": "Apply fixes from step-1", "depends_on": ["step-1"]}
  ]
}
```

Steps execute in topological order. Output from earlier steps flows as input to later steps. Use `--dry-run` to inspect the plan before execution.

## Pre-flight Validation

Before any plan executes, four validation passes run:

| Pass | Detects |
|------|---------|
| Capability check | Handler unavailable, missing API keys, disabled feature flags |
| Input validation | Malformed inputs, missing fields, invalid step references |
| Risk assessment | Unconfirmed security actions, high-risk steps without safeguards |
| Dependency check | Circular dependencies, unreachable steps, broken references |

If any pass produces a CRITICAL flag, the plan is blocked and the flags are returned to the user. Volition does not auto-remediate CRITICAL flags (Constitution Rule 3).

## Fallback Chains

When a handler fails, Volition attempts fallbacks before giving up:

- `code_edit` falls back to `text_edit`
- Other handlers have no fallbacks (failure is reported immediately)

If all fallbacks are exhausted, Volition stops executing any remaining steps in the plan; previously completed steps are not rolled back (Constitution Rule 6).

## Security Constraints (R.22-R.23)

### Shodan Queries

Shodan queries are **restricted operations** with mandatory safeguards:

1. **Explicit --confirm Flag Required (R.22.1)**: Every Shodan query requires `--confirm` to execute. Without it, the query returns a `confirmation_required` status with guidance:
   ```bash
   python3 volition.py query security "exposed MongoDB instances" --confirm
   ```

2. **Disable Flag (R.22.2)**: Disable Shodan via environment:
   ```bash
   VOLITION_DISABLE_SHODAN=true
   NOCP_FLAG_SHODAN_ENABLED=false
   ```

3. **Data Redaction (R.22.3)**: IP addresses in results are partially redacted (e.g., `192.168.xxx.xxx`).

4. **Audit Logging with Severity (R.22.4)**: All queries are logged to `~/.volition/shodan_audit.jsonl` with fields: `requester`, `query`, `target`, `result_count`, `severity`, and `timestamp`.

5. **Rate Limiting (R.23.1)**: Maximum 10 queries per hour. Exceeding this limit returns an error with the remaining cooldown time.

6. **Clear Error on Unconfirmed (R.23.2)**: Unconfirmed queries return explicit guidance.

## Configuration

### Required Environment Variables

Set in `.env.local`:

```bash
# Code Editing (Serena) — no API key required, uses local LSP servers

# LLM Consultation (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...

# Web Search
OPENAI_API_KEY=sk-...  # Shared with LLM

# Security Queries
SHODAN_API_KEY=...  # Optional, enables security queries
```

### Tuning Variables

```bash
VOLITION_CONFIDENCE_THRESHOLD=0.65   # Classification confidence gate
VOLITION_EMBEDDING_WEIGHT=0.7        # Weight for embedding signal in RRF
VOLITION_KEYWORD_WEIGHT=0.3          # Weight for keyword signal in RRF
VOLITION_LEARNING_RATE=0.2           # Feedback adjustment sensitivity
```

## Backends

Volition uses these libraries internally (implementation details are **opaque**):

- **Serena**: LSP-powered semantic code editing
- **cross-llm**: Multi-provider LLM access
- **openai-websearch**: Web search with reasoning
- **shodan-client**: Security reconnaissance

You interact only through Volition's unified command interface. The underlying protocols (MCP, LSP, HTTP APIs) are hidden.

## Error Handling

Volition returns structured errors without exposing internal details:

```json
{"status": "error", "code": 2, "message": "Symbol not found: UserAuth.validate"}
```

Error codes:
- `1`: Invalid input
- `2`: Resource not found
- `3`: Backend unavailable
- `4`: Permission denied (e.g., unconfirmed security query)

## Synergies

Volition integrates with other Cognitive Construct skills:

- **→ Inland Empire**: Completed actions are logged as memories for future recall
- **← Rhetoric**: When `RHETORIC_PREFLIGHT` is enabled, high-risk steps trigger a Rhetoric deliberation before execution
- **← Encyclopedia**: Code edits can fetch documentation context

Synergies operate transparently — if a downstream skill is unavailable (e.g., `shared.synergies` import fails), Volition continues without error. No feature flag is required; synergies are always attempted when the corresponding module is available.

## Constitution

Seven inviolable rules govern Volition's behavior. See `constitution.md` for full text.

1. **Never execute** when the classifier is not confident — ask for clarification
2. **Never execute** a security action without `--confirm`
3. **Never execute** a plan that fails pre-flight validation
4. **Never hide** classification uncertainty from the user
5. **Never let** feedback adjustment override safety constraints
6. **Always abort** the entire plan on step failure
7. **Always log** before acting (audit entry written before execution begins)

## Files

- `~/.volition/audit.log`: General audit trail (JSONL, append-only)
- `~/.volition/shodan_audit.jsonl`: Shodan-specific audit log with severity metadata
- `~/.volition/rate_limits.json`: Rate limit state

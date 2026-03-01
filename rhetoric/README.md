# Rhetoric: Deliberation + Argumentation Engine

> "Give me a place to stand, and I will move the world."

Rhetoric has two subsystems: **deliberation** (multi-model dialectical debate) and **validation** (Toulmin argumentation engine with symbolic verification).

## Quick Start

```bash
# Check system status
python3 rhetoric/scripts/rhetoric.py status

# Deliberate on a question (requires 2+ model API keys)
python3 rhetoric/scripts/rhetoric.py deliberate "Should we use microservices or monolith?" --rounds 2

# Run Toulmin validation demos (no API needed)
python3 rhetoric/scripts/rhetoric.py demo

# Analyze argument structure (requires INCEPTION_API_KEY)
python3 rhetoric/scripts/rhetoric.py plan "PostgreSQL 16 improves performance for most workloads"

# Validate a pre-built argument graph (no API needed)
python3 rhetoric/scripts/rhetoric.py validate examples/graph.json
```

## Commands

| Command | Subsystem | API Required | Description |
|---------|-----------|--------------|-------------|
| `deliberate` | ai-counsel | 2+ LLM keys | Multi-model dialectical debate |
| `status` | ai-counsel | any LLM key | Show available providers |
| `plan` | toulmin | INCEPTION_API_KEY | Decompose + validate + strategize |
| `validate` | toulmin | none | Validate a JSON argument graph |
| `demo` | toulmin | none | Run built-in demo cases |

## Directory Structure

```text
rhetoric/
├── SKILL.md                  # Skill metadata (v4.0.0)
├── constitution.md           # 6 inviolable rules
├── README.md
├── .env.example
└── scripts/
    ├── rhetoric.py           # Unified CLI entrypoint
    ├── ai-counsel/           # Deliberation engine
    │   ├── ai_counsel/       # Client library
    │   ├── adapters/         # HTTP adapters (OpenAI, Anthropic, etc.)
    │   ├── deliberation/     # Engine with dialectical roles
    │   └── models/           # Pydantic schemas
    └── toulmin/              # Toulmin validation engine
        ├── models.py         # Domain types (ArgumentGraph, Claim, etc.)
        ├── validate.py       # Four-pass symbolic validation
        ├── decompose.py      # LLM-powered intent → graph decomposition
        ├── strategy.py       # Audience-based delivery strategy
        ├── bridge.py         # Structural analogy via Mercury
        ├── engine.py         # Pipeline orchestrator
        ├── cli.py            # Exported functions for rhetoric.py
        └── tests/            # 39 deterministic tests
```

## Toulmin Validation Passes

The engine runs four passes on every argument graph — pure functions, no LLM calls:

1. **Structural completeness** — Required nodes (claim, data, warrant) + recommended (backing, qualifier, rebuttals)
2. **Inferential type validation** — Formal fallacies (affirming consequent), hasty generalization, false cause, surface analogy
3. **Cross-reference integrity** — Contradicting evidence not addressed, unused supporting evidence
4. **Qualifier calibration** — Claim strength vs. evidence strength mismatch

## Dialectical Roles

| # | Role | Produces |
|---|------|----------|
| 1 | **Proponent** | Claim, Warrant, Evidence, Anticipated Objections |
| 2 | **Opponent** | Counter-Claim, Warrant, Evidence, Rebuttal |
| 3+ | **Synthesizer** | Agreement, Tensions, Resolution, Remaining Risks |

## Configuration

### Required Environment Variables

**Deliberation** — at least 2 of:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
OLLAMA_URL=http://localhost:11434
```

**Toulmin plan** (optional):
```bash
INCEPTION_API_KEY=...   # Mercury API for decomposition + analogy bridge
```

## Dependencies

- Python >= 3.12
- httpx
- pydantic
- tenacity (for retry logic in ai-counsel)

## License

MIT

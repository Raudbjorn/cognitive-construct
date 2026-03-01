# Rhetoric: Toulmin Argumentation Engine

> "Give me a place to stand, and I will move the world."

Symbolic argumentation engine based on Stephen Toulmin's model. Decomposes arguments into formal graphs, validates inferential steps, and calibrates delivery to audience epistemic state.

## Quick Start

```bash
# Run Toulmin validation demos (no API needed)
python3 rhetoric/scripts/rhetoric.py demo

# Validate a pre-built argument graph (no API needed)
python3 rhetoric/scripts/rhetoric.py validate examples/graph.json

# Analyze argument structure (requires INCEPTION_API_KEY)
python3 rhetoric/scripts/rhetoric.py plan "PostgreSQL 16 improves performance for most workloads"
```

## Commands

| Command | API Required | Description |
|---------|--------------|-------------|
| `plan` | INCEPTION_API_KEY | Decompose + validate + strategize |
| `validate` | none | Validate a JSON argument graph |
| `demo` | none | Run built-in demo cases |

## Directory Structure

```text
rhetoric/
├── SKILL.md                  # Skill metadata (v5.0.0)
├── constitution.md           # 6 inviolable rules
├── README.md
├── .env.example
└── scripts/
    ├── rhetoric.py           # CLI entrypoint
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

## Validation Passes

The engine runs four passes on every argument graph — pure functions, no LLM calls:

1. **Structural completeness** — Required nodes (claim, data, warrant) + recommended (backing, qualifier, rebuttals)
2. **Inferential type validation** — Formal fallacies (affirming consequent), hasty generalization, false cause, surface analogy
3. **Cross-reference integrity** — Contradicting evidence not addressed, unused supporting evidence
4. **Qualifier calibration** — Claim strength vs. evidence strength mismatch

## Configuration

```bash
# Only needed for 'plan' command
INCEPTION_API_KEY=...   # Mercury API for decomposition + analogy bridge
```

The `validate` and `demo` commands require no API keys or network access.

## Dependencies

- Python >= 3.12
- httpx
- pydantic

## License

MIT

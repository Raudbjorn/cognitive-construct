---
name: rhetoric
description: Toulmin argumentation engine — symbolic validation of argument structure, inferential integrity, and audience-calibrated delivery.
license: MIT
metadata:
  version: 5.0.0
  dependencies: python>=3.12, httpx, pydantic
---

# Rhetoric: Toulmin Argumentation Engine

> "Give me a place to stand, and I will move the world." — Archimedes

## Overview

**Rhetoric** is a symbolic argumentation engine based on Stephen Toulmin's model. It decomposes natural-language arguments into formal graphs (claim, data, warrant, backing, qualifier, rebuttal), validates inferential steps through four deterministic passes, selects delivery strategy based on audience epistemic state, and optionally generates structural analogies via Mercury.

## Commands

### `plan "<intent>" [--no-bridge] [--contradict <evidence>...]`
Decompose a natural-language argument into a Toulmin graph, validate it through four passes, select a delivery strategy, and optionally generate a structural analogy via Mercury.

```bash
python3 scripts/rhetoric.py plan "PostgreSQL 16 improves performance for most workloads"
python3 scripts/rhetoric.py plan "Rust is best for all backends" --contradict "Compile times are slow" "Python has larger ecosystem"
python3 scripts/rhetoric.py plan "Microservices improve reliability" --no-bridge
```

**Requires:** `INCEPTION_API_KEY` environment variable (Mercury API)

**Options:**
- `--no-bridge`: Skip analogy bridge generation
- `--contradict <evidence>...`: Known contradicting evidence for cross-reference validation

### `validate <file>`
Validate a pre-built Toulmin argument graph from a JSON file. No API needed.

```bash
python3 scripts/rhetoric.py validate examples/hasty-generalization.json
```

### `demo`
Run built-in demonstration cases showcasing the validation engine. No API needed.

```bash
python3 scripts/rhetoric.py demo
```

Demonstrates detection of: valid induction, hasty generalization, formal fallacies (affirming the consequent), cherry-picking (cross-reference integrity), and surface analogies.

## Validation Passes

The engine runs four deterministic validation passes (no LLM calls):

| Pass | Name | Detects |
|------|------|---------|
| 1 | Structural completeness | Missing claim/data/warrant, absent backing/qualifier/rebuttals |
| 2 | Inferential type validation | Formal fallacies, hasty generalization, false cause, surface analogy |
| 3 | Cross-reference integrity | Unaddressed contradictions, unused supporting evidence |
| 4 | Qualifier calibration | Overclaim/underclaim relative to evidence strength |

## Required Credentials

```bash
INCEPTION_API_KEY=...   # Mercury API — only needed for 'plan' command
```

The `validate` and `demo` commands require no API keys.

## Constitution

See [constitution.md](constitution.md) for 6 inviolable rules governing the argumentation engine.

## Synergies

- **-> Encyclopedia**: Provide contradicting/supporting evidence for validation Pass 3
- **-> Inland Empire**: Store significant argument analyses as memories
- **<- Volition**: Complex actions can request argument validation first

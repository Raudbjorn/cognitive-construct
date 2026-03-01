---
name: rhetoric
description: Dialectical deliberation and Toulmin argumentation engine — two subsystems for structured reasoning.
license: MIT
metadata:
  version: 4.0.0
  dependencies: python>=3.12, httpx, pydantic
---

# Rhetoric: Deliberation + Argumentation Engine

> "Give me a place to stand, and I will move the world." — Archimedes

## Overview

**Rhetoric** has two complementary subsystems:

1. **Deliberation** (ai-counsel) — Multi-model dialectical debate with convergence detection. Models argue as proponent/opponent/synthesizer.
2. **Validation** (toulmin) — Symbolic Toulmin argumentation engine that decomposes arguments into formal graphs, validates inferential steps, and calibrates delivery to audience epistemic state.

## Commands

### Deliberation Subsystem

#### `deliberate "<question>" [--rounds <n>]`
Deliberate on a question through structured multi-model dialectic.

```bash
python3 scripts/rhetoric.py deliberate "Should we use microservices or monolith?" --rounds 3
```

**Output:**
```json
{
  "status": "completed",
  "question": "Should we use microservices or monolith?",
  "rounds_completed": 3,
  "consensus": "Monolith recommended for MVP, with clear module boundaries for future extraction",
  "confidence": 0.85
}
```

**Options:**
- `--rounds <n>`: Number of deliberation rounds (1-5, default: 2)
- `--context <text>`: Additional context for deliberation
- `--debug`: Show role assignments and internal deliberation details
- `--allow-single`: Allow single model for development testing

**Note:** Requires at least 2 configured model credentials.

#### `status`
Get current system status including available providers and deliberation readiness.

```bash
python3 scripts/rhetoric.py status
```

### Toulmin Validation Subsystem

#### `plan "<intent>" [--no-bridge] [--contradict <evidence>...]`
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

#### `validate <file>`
Validate a pre-built Toulmin argument graph from a JSON file. No API needed.

```bash
python3 scripts/rhetoric.py validate examples/hasty-generalization.json
```

#### `demo`
Run built-in demonstration cases showcasing the validation engine. No API needed.

```bash
python3 scripts/rhetoric.py demo
```

Demonstrates detection of: valid induction, hasty generalization, formal fallacies (affirming the consequent), cherry-picking (cross-reference integrity), and surface analogies.

## Validation Passes

The Toulmin engine runs four deterministic validation passes (no LLM calls):

| Pass | Name | Detects |
|------|------|---------|
| 1 | Structural completeness | Missing claim/data/warrant, absent backing/qualifier/rebuttals |
| 2 | Inferential type validation | Formal fallacies, hasty generalization, false cause, surface analogy |
| 3 | Cross-reference integrity | Unaddressed contradictions, unused supporting evidence |
| 4 | Qualifier calibration | Overclaim/underclaim relative to evidence strength |

## Dialectical Roles

Participants are automatically assigned roles in order:

| Position | Role | Function |
|----------|------|----------|
| 1st | **Proponent** | Argues in favor of the strongest position |
| 2nd | **Opponent** | Challenges assumptions, finds weaknesses |
| 3rd+ | **Synthesizer** | Integrates positions, proposes resolution |

## Required Credentials

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

## Constitution

See [constitution.md](constitution.md) for 6 inviolable rules governing both subsystems.

## Synergies

- **-> Encyclopedia**: Fetch context during deliberation; provide contradicting/supporting evidence for validation Pass 3
- **-> Inland Empire**: Store significant decisions as memories
- **<- Volition**: Complex actions can request deliberation or validation first

# Rhetoric-Rejected Components

Components removed from `rhetoric/` because they are not rhetoric.

## What's Here

### `sequentialthinking_py/`
**Thought journaling** — recording and branching internal monologue. This is *noesis* (thinking), not *rhetoric* (argument). Sequential thinking tracks chains of reasoning; it doesn't engage in persuasion, debate, or dialectical exchange.

### `vibecheck_py/`
**Metacognitive pattern detection** — asking "am I biased?" This is *introspection*, not *persuasion or debate*. VibeCheck identifies cognitive patterns and risks; it doesn't construct or contest arguments.

### `references/constitution.md`
Behavioral guardrails for VibeCheck sessions. Safety rules and constraints, not argument structure or deliberation format.

### `references/reflection_template.md`
Self-reflection log template. An introspection artifact for recording risk assessments and constitution checks — not related to argumentative deliberation.

## Why They Were Removed

Rhetoric conflated three distinct concerns:

1. **Noesis** (thinking/journaling) — `sequentialthinking_py`
2. **Introspection** (metacognition/bias detection) — `vibecheck_py`
3. **Rhetoric** (argumentation/deliberation) — `ai-counsel`

Only the third is actual rhetoric. Keeping the others in `rhetoric/` diluted the skill's conceptual integrity and created a grab-bag of loosely related cognitive tools rather than a focused deliberation engine.

## Can These Be Reused?

Yes. These are functional, tested components. They could become standalone skills or be integrated into other parts of Cognitive Construct:

- `sequentialthinking_py` → a dedicated `noesis` or `thinking` skill
- `vibecheck_py` → a dedicated `introspection` or `metacognition` skill

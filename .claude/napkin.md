# Napkin

## Corrections

| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-01 | self | `require_skill_config("rhetoric")` runs for ALL commands, blocking toulmin commands that don't need LLM keys | Scope config checks to the commands that actually need them |
| 2026-03-01 | self | Ran CLI smoke tests with wrong path (cwd was inside ai-counsel dir) | Always use absolute paths or verify cwd before running CLI scripts |

## User Preferences
- Liberal emoji usage (per CLAUDE.md)
- Prefers `sed` for bulk import rewrites over manual editing (faster, less error-prone)
- Skills are organized as `skill_name/scripts/package_name/` with `sys.path.insert` for local imports
- Deprecated/removed code goes to `cognitive-construct-kb/` repo, not just deleted — with README explaining why

## Patterns That Work
- `sed 's/from old\./from new./g' src > dst` for bulk import rewrites across a package migration — verified with `grep -rn` after
- Lazy imports in CLI dispatch (`from toulmin.cli import run_plan`) to avoid loading heavy deps for unrelated commands
- Running PoC tests first (`PYTHONPATH=. python -m pytest`) to confirm baseline before migrating
- Creating `__init__.py` with selective public exports — keeps the API surface clean
- Scoping PRs: stage only relevant files when working tree has mixed changes from multiple concerns

## Patterns That Don't Work
- (none yet)

## Domain Notes
- `rhetoric/` is now pure Toulmin argumentation (v5.0.0) — ai-counsel was moved to cognitive-construct-kb
- Dialectic (truth through debate) ≠ Rhetoric (structured argumentation) — these are different Aristotelian traditions
- Toulmin package uses Pydantic models (justified exception to dataclass preference) for `model_copy()`, `model_validate()`, `Field(min_length=1)`
- `constitution.md` governs the argumentation engine — 6 inviolable rules from spec §4
- PR #6 on feature/toulmin-validation-engine: Toulmin integration + ai-counsel removal

# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-01 | self | `require_skill_config("rhetoric")` runs for ALL commands, blocking toulmin commands that don't need LLM keys | Scope config checks to the commands that actually need them — wrap in `if args.command in ("deliberate", "status")` |
| 2026-03-01 | self | Ran CLI smoke tests with wrong path (`rhetoric/scripts/rhetoric.py` doubled the path from cwd) | Always use absolute paths or verify cwd before running CLI scripts |

## User Preferences
- Liberal emoji usage (per CLAUDE.md)
- Prefers `sed` for bulk import rewrites over manual editing (faster, less error-prone)
- Skills are organized as `skill_name/scripts/package_name/` with `sys.path.insert` for local imports

## Patterns That Work
- `sed 's/from old\./from new./g' src > dst` for bulk import rewrites across a package migration — verified with `grep -rn` after
- Lazy imports in CLI dispatch (`from toulmin.cli import run_plan`) to avoid loading heavy deps (pydantic, httpx) for unrelated commands
- Running PoC tests first (`PYTHONPATH=. python -m pytest`) to confirm baseline before migrating
- Creating `__init__.py` with selective public exports — keeps the API surface clean

## Patterns That Don't Work
- (none yet)

## Domain Notes
- `rhetoric/` has two subsystems: `ai-counsel` (deliberation) and `toulmin` (validation) — they share `rhetoric.py` as entrypoint
- `shared/config_check.py` provides `require_skill_config()` which gates on LLM provider availability — not all commands need it
- Toulmin package uses Pydantic models (justified exception to dataclass preference) for `model_copy()`, `model_validate()`, `Field(min_length=1)`
- ai-counsel has 577 passing unit tests + 4 pre-existing failures (vcrpy missing, decision graph integration) — unrelated to toulmin work
- `constitution.md` governs both subsystems — 6 inviolable rules from spec §4

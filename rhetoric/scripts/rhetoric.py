#!/usr/bin/env python3
"""
Rhetoric: Deliberation + Argumentation Engine
CLI entrypoint for dialectical deliberation and Toulmin argument validation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add parent directories to path for local module imports
# Structure: rhetoric/scripts/rhetoric.py, rhetoric/scripts/ai-counsel/, rhetoric/scripts/toulmin/
_scripts_dir = Path(__file__).parent
_rhetoric_dir = _scripts_dir.parent
sys.path.insert(0, str(_scripts_dir / "ai-counsel"))
sys.path.insert(0, str(_scripts_dir))  # Makes 'import toulmin' work

from ai_counsel.client import AICounselClient, Participant
from ai_counsel.types import DeliberationResult

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger("rhetoric")

# Dialectical role assignment order
_ROLE_ORDER = ("proponent", "opponent", "synthesizer")


def _assign_roles(participants: List[Participant]) -> List[Participant]:
    """Auto-assign dialectical roles to participants.

    First = proponent, second = opponent, third+ = synthesizer.
    """
    assigned = []
    for i, p in enumerate(participants):
        role = _ROLE_ORDER[min(i, len(_ROLE_ORDER) - 1)]
        assigned.append(Participant(adapter=p.adapter, model=p.model, role=role))
    return assigned


async def cmd_deliberate(args: argparse.Namespace) -> None:
    """Execute deliberate command with dialectical role assignment."""
    # Check for API keys and build available models list
    available_models: List[str] = []
    if os.environ.get("OPENAI_API_KEY"):
        available_models.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_models.append("anthropic")
    if os.environ.get("OPENROUTER_API_KEY"):
        available_models.append("openrouter")
    if os.environ.get("OLLAMA_URL"):
        available_models.append("ollama")

    if len(available_models) < 2 and not args.allow_single:
        print(json.dumps({
            "status": "error",
            "code": 4,
            "message": "Insufficient models for deliberation. Configure at least 2 of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OLLAMA_URL"
        }, indent=2))
        sys.exit(1)

    # Setup participants based on available keys
    participants: List[Participant] = []
    if "openai" in available_models:
        participants.append(Participant(adapter="openai", model="gpt-4o"))
    if "anthropic" in available_models:
        participants.append(Participant(adapter="anthropic", model="claude-sonnet-4-20250514"))
    if "openrouter" in available_models and len(participants) < 3:
        participants.append(Participant(adapter="openrouter", model="google/gemini-flash-1.5"))
    if "ollama" in available_models and len(participants) < 2:
        participants.append(Participant(adapter="ollama", model="llama3.2"))

    if len(participants) < 2:
        print(json.dumps({
            "status": "error",
            "code": 4,
            "message": "Could not configure 2 participants for deliberation"
        }, indent=2))
        sys.exit(1)

    # Auto-assign dialectical roles
    participants = _assign_roles(participants)

    if args.debug:
        role_summary = ", ".join(
            f"{p.model}@{p.adapter}={p.role}" for p in participants
        )
        print(f"[debug] Roles: {role_summary}", file=sys.stderr)

    client = AICounselClient(enable_transcripts=True, enable_decision_graph=False)

    try:
        result = await client.deliberate(
            question=args.question,
            participants=participants,
            rounds=args.rounds,
            context=args.context
        )

        if result.is_ok():
            val = result.value
            # Calculate confidence from convergence info if available
            confidence = 0.85
            if val.convergence_info and val.convergence_info.final_similarity:
                confidence = val.convergence_info.final_similarity

            print(json.dumps({
                "status": "completed",
                "question": args.question,
                "rounds_completed": val.rounds_completed,
                "consensus": val.summary.consensus,
                "confidence": round(confidence, 2)
            }, indent=2))
        else:
            print(json.dumps({
                "status": "error",
                "message": result.error.message
            }, indent=2))
            sys.exit(1)
    finally:
        await client.close()


async def cmd_status(args: argparse.Namespace) -> None:
    """Execute status command — report providers and deliberation readiness."""
    available_providers: List[str] = []
    if os.environ.get("OPENAI_API_KEY"):
        available_providers.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
    if os.environ.get("OPENROUTER_API_KEY"):
        available_providers.append("openrouter")
    if os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        available_providers.append("gemini")
    if os.environ.get("OLLAMA_URL"):
        available_providers.append("ollama")

    models = len(available_providers)

    print(json.dumps({
        "models_available": models,
        "providers": available_providers,
        "deliberation_ready": models >= 2,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rhetoric: Deliberation + Argumentation Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # --- Deliberation subsystem (ai-counsel) ---

    deliberate_parser = subparsers.add_parser("deliberate", help="Deliberate on a question")
    deliberate_parser.add_argument("question", help="Question to deliberate")
    deliberate_parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    deliberate_parser.add_argument("--context", help="Context for deliberation")
    deliberate_parser.add_argument("--debug", action="store_true", help="Show debug info")
    deliberate_parser.add_argument("--allow-single", action="store_true", help="Allow single model (dev)")

    subparsers.add_parser("status", help="Get system status")

    # --- Toulmin validation subsystem ---

    plan_parser = subparsers.add_parser("plan", help="Analyze argument structure (Toulmin)")
    plan_parser.add_argument("intent", help="The argument to analyze")
    plan_parser.add_argument("--no-bridge", action="store_true", help="Skip analogy bridge")
    plan_parser.add_argument("--contradict", nargs="*", default=[], help="Known contradicting evidence")

    validate_parser = subparsers.add_parser("validate", help="Validate a Toulmin graph from JSON")
    validate_parser.add_argument("file", help="Path to JSON ArgumentGraph file")

    subparsers.add_parser("demo", help="Run Toulmin validation demos (no API needed)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check configuration before running deliberation commands
    if args.command in ("deliberate", "status"):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from shared.config_check import require_skill_config
            require_skill_config("rhetoric")
        except ImportError:
            pass  # shared module not available, skip check

    if args.command == "deliberate":
        asyncio.run(cmd_deliberate(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "plan":
        from toulmin.cli import run_plan
        asyncio.run(run_plan(args))
    elif args.command == "validate":
        from toulmin.cli import run_validate
        run_validate(args)
    elif args.command == "demo":
        from toulmin.cli import run_demo
        run_demo()


if __name__ == "__main__":
    main()

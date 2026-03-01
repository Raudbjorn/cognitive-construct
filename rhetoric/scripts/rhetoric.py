#!/usr/bin/env python3
"""
Rhetoric: Toulmin Argumentation Engine
CLI entrypoint for symbolic argument validation and delivery strategy.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add scripts dir to path for local module imports
# Structure: rhetoric/scripts/rhetoric.py, rhetoric/scripts/toulmin/
_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))  # Makes 'import toulmin' work


def main() -> None:
    parser = argparse.ArgumentParser(description="Rhetoric: Toulmin Argumentation Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

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

    if args.command == "plan":
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

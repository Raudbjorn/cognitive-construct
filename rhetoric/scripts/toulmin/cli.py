"""Toulmin validation engine — exported functions for rhetoric.py integration.

Standalone usage (dev):
    cd rhetoric/scripts && PYTHONPATH=. python -m toulmin.cli

Integrated usage (via rhetoric.py):
    python3 rhetoric/scripts/rhetoric.py plan "Microservices are better than monoliths"
    python3 rhetoric/scripts/rhetoric.py validate examples/graph.json
    python3 rhetoric/scripts/rhetoric.py demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from toulmin.engine import EngineError, RhetoricEngine, RhetoricPlan
from toulmin.models import (
    ArgumentGraph,
    Backing,
    BackingSource,
    Claim,
    ClaimScope,
    Confidence,
    DataSource,
    DataStrength,
    Datum,
    InferenceType,
    Qualifier,
    Rebuttal,
    RebuttalSeverity,
    Warrant,
)
from toulmin.validate import validate


async def run_plan(args: argparse.Namespace) -> None:
    """Run the full Toulmin pipeline: decompose → validate → strategize → bridge."""
    api_key = os.environ.get("INCEPTION_API_KEY", "")
    if not api_key:
        print("ERROR: INCEPTION_API_KEY environment variable not set", file=sys.stderr)
        print("  Set it with: export INCEPTION_API_KEY='your-key-here'", file=sys.stderr)
        sys.exit(1)

    engine = RhetoricEngine(
        api_key=api_key,
        enable_bridge=not args.no_bridge,
    )

    contradictions: list[str] = args.contradict or []

    result = await engine.plan(
        args.intent,
        known_contradictions=contradictions if contradictions else None,
    )

    if isinstance(result, EngineError):
        print(f"\n❌ Pipeline failed at stage: {result.stage}")
        print(f"   {result.message}")
        if result.details:
            print(f"   Details: {result.details[:200]}")
        sys.exit(2)

    print_plan(result)


def run_validate(args: argparse.Namespace) -> None:
    """Validate a Toulmin argument graph from a JSON file."""
    try:
        with open(args.file) as f:
            data = json.load(f)
        graph = ArgumentGraph.model_validate(data)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, Exception) as e:
        print(f"ERROR: Invalid graph file: {e}", file=sys.stderr)
        sys.exit(1)

    result = validate(graph)
    print(f"\nValidation: {result.status.value}")
    print(f"Flags: {len(result.flags)} ({len(result.critical_flags)} critical)\n")

    for flag in result.flags:
        severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        icon = severity_icon.get(flag.severity.value, "⚪")
        print(f"  {icon} [{flag.type}] @ {flag.location}")
        print(f"    {flag.description}")
        print(f"    → {flag.remediation}\n")


def run_demo() -> None:
    """Run built-in demonstration cases showing the engine catching bad arguments."""
    print("=" * 70)
    print("RHETORIC ENGINE — DEMONSTRATION")
    print("Formal verification of informal discourse")
    print("=" * 70)

    cases = build_demo_cases()

    for i, (name, graph, contradictions, description) in enumerate(cases, 1):
        print(f"\n{'─' * 70}")
        print(f"CASE {i}: {name}")
        print(f"{'─' * 70}")
        print(f"  {description}\n")
        print(f"  Claim: \"{graph.claim.text}\"")
        print(f"  Scope: {graph.claim.scope.value} | Confidence: {graph.claim.confidence.value}")
        print(f"  Evidence: {len(graph.data)} pieces")
        print(f"  Warrant: \"{graph.warrant.text}\"")
        print(f"  Inference type: {graph.warrant.inference_type.value}")
        if graph.qualifier:
            print(f"  Qualifier: {graph.qualifier.strength.value}")
        print()

        result = validate(
            graph,
            known_contradictions=contradictions,
        )

        status_icon = {
            "valid": "✅",
            "flagged": "⚠️ ",
            "invalid": "❌",
            "pending": "⏳",
        }
        print(f"  Result: {status_icon.get(result.status.value, '?')} {result.status.value.upper()}")
        print(f"  Flags: {len(result.flags)} total, {len(result.critical_flags)} critical")

        if result.qualifier_calibration:
            cal = result.qualifier_calibration
            if not cal.calibrated:
                print(f"  Qualifier adjustment: {cal.current.value} → {cal.recommended.value}")

        for flag in result.flags:
            severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
            icon = severity_icon.get(flag.severity.value, "⚪")
            print(f"\n    {icon} {flag.type}")
            print(f"      {flag.description}")
            print(f"      Fix: {flag.remediation}")

    print(f"\n{'=' * 70}")
    print("END OF DEMONSTRATION")
    print(f"{'=' * 70}")


def build_demo_cases() -> list[tuple[str, ArgumentGraph, list[str] | None, str]]:
    """Build the demonstration cases.

    Each case is a (name, graph, contradictions, description) tuple.
    The cases are designed to showcase specific validation capabilities.
    """
    return [
        # Case 1: Valid induction
        (
            "Valid Induction (should PASS)",
            ArgumentGraph(
                claim=Claim(
                    text="PostgreSQL 16 likely improves query performance for most workloads",
                    scope=ClaimScope.GENERAL,
                    confidence=Confidence.PROBABLE,
                ),
                data=[
                    Datum(
                        text="Performance benchmarks across n=2400 production databases showed 87% experienced faster queries after upgrading to PostgreSQL 16",
                        source=DataSource.ENCYCLOPEDIA,
                        strength=DataStrength.STRONG,
                    ),
                ],
                warrant=Warrant(
                    text="Large-scale benchmarks on production systems generalize to typical production workloads",
                    inference_type=InferenceType.INDUCTIVE,
                ),
                backing=Backing(
                    text="The benchmarks were conducted by the PostgreSQL Performance Team using representative workload profiles",
                    source=BackingSource.EMPIRICAL_RESEARCH,
                ),
                qualifier=Qualifier(
                    text="in most cases",
                    strength=Confidence.PROBABLE,
                ),
                rebuttals=[
                    Rebuttal(
                        text="Workloads with heavy write contention may not see improvement",
                        severity=RebuttalSeverity.SIGNIFICANT,
                        addressed=True,
                        response="The benchmarks included write-heavy profiles; improvement was smaller (62%) but still present",
                    ),
                ],
            ),
            None,
            "Well-structured inductive argument with strong evidence, calibrated qualifier, and addressed rebuttal.",
        ),

        # Case 2: Hasty generalization
        (
            "Hasty Generalization (should FLAG)",
            ArgumentGraph(
                claim=Claim(
                    text="Upgrading to PostgreSQL 16 always improves performance",
                    scope=ClaimScope.UNIVERSAL,
                    confidence=Confidence.CERTAIN,
                ),
                data=[
                    Datum(
                        text="3 blog posts reported improved performance after upgrading to PostgreSQL 16",
                        source=DataSource.INFERRED,
                        strength=DataStrength.ANECDOTAL,
                    ),
                ],
                warrant=Warrant(
                    text="Blog reports of performance improvements indicate universal improvement",
                    inference_type=InferenceType.INDUCTIVE,
                ),
                qualifier=Qualifier(
                    text="always",
                    strength=Confidence.CERTAIN,
                ),
            ),
            None,
            "Same topic as Case 1, but with anecdotal evidence overclaimed as universal truth.\n  The engine should catch: hasty generalization, overclaim, missing backing, missing rebuttals.",
        ),

        # Case 3: Formal fallacy (affirming the consequent)
        (
            "Formal Fallacy — Affirming the Consequent (should INVALIDATE)",
            ArgumentGraph(
                claim=Claim(
                    text="The server must be running Linux",
                    scope=ClaimScope.PARTICULAR,
                    confidence=Confidence.CERTAIN,
                ),
                data=[
                    Datum(
                        text="The server responds to SSH connections on port 22",
                        source=DataSource.USER_STATED,
                        strength=DataStrength.STRONG,
                    ),
                ],
                warrant=Warrant(
                    text="Linux servers run SSH, and this server runs SSH, so it must be Linux",
                    inference_type=InferenceType.DEDUCTIVE,
                    formalization="B, A→B ⊢ A",
                ),
                backing=Backing(
                    text="Most Linux distributions include OpenSSH by default",
                    source=BackingSource.DOMAIN_EXPERTISE,
                ),
                qualifier=Qualifier(
                    text="necessarily",
                    strength=Confidence.CERTAIN,
                ),
            ),
            None,
            "Classic formal fallacy: 'If Linux then SSH; SSH observed; therefore Linux.'\n  Ignores that BSD, macOS, and Windows also run SSH.",
        ),

        # Case 4: Cross-reference integrity failure
        (
            "Cherry-Picking (should FLAG integrity)",
            ArgumentGraph(
                claim=Claim(
                    text="Rust is the best language for all backend services",
                    scope=ClaimScope.UNIVERSAL,
                    confidence=Confidence.PROBABLE,
                ),
                data=[
                    Datum(
                        text="Rust has zero-cost abstractions and memory safety without garbage collection",
                        source=DataSource.ENCYCLOPEDIA,
                        strength=DataStrength.STRONG,
                    ),
                    Datum(
                        text="Discord rewrote their Read States service from Go to Rust with significant latency improvements",
                        source=DataSource.ENCYCLOPEDIA,
                        strength=DataStrength.MODERATE,
                    ),
                ],
                warrant=Warrant(
                    text="Performance and safety advantages make Rust superior for all backend use cases",
                    inference_type=InferenceType.INDUCTIVE,
                ),
                backing=Backing(
                    text="Systems programming languages with memory safety provide the best foundation",
                    source=BackingSource.DOMAIN_EXPERTISE,
                ),
                qualifier=Qualifier(
                    text="in most cases",
                    strength=Confidence.PROBABLE,
                ),
                rebuttals=[
                    Rebuttal(
                        text="Rust has a steep learning curve",
                        severity=RebuttalSeverity.SIGNIFICANT,
                        addressed=True,
                        response="The long-term productivity gains outweigh initial learning costs",
                    ),
                ],
            ),
            [
                "Rust compile times are significantly longer than Go or Java, impacting development velocity",
                "Python and TypeScript dominate in rapid prototyping and have larger ecosystems for web services",
            ],
            "The argument ignores known evidence from the Encyclopedia that complicates the claim.\n  The engine should catch: incomplete rebuttal (contradicting evidence not addressed).",
        ),

        # Case 5: Surface analogy
        (
            "Surface Analogy (should FLAG)",
            ArgumentGraph(
                claim=Claim(
                    text="Microservices improve system reliability",
                    scope=ClaimScope.GENERAL,
                    confidence=Confidence.PLAUSIBLE,
                ),
                data=[
                    Datum(
                        text="Netflix successfully uses microservices at scale",
                        source=DataSource.ENCYCLOPEDIA,
                        strength=DataStrength.MODERATE,
                    ),
                ],
                warrant=Warrant(
                    text="Microservices look like biological cells — independent units that together form a resilient organism",
                    inference_type=InferenceType.ANALOGICAL,
                ),
                qualifier=Qualifier(
                    text="plausibly",
                    strength=Confidence.PLAUSIBLE,
                ),
            ),
            None,
            "The warrant uses a biological analogy, but it's surface-level (looks like)\n  rather than structural (functions as). Cells and microservices share appearance\n  but not the same failure/recovery mechanisms.",
        ),
    ]


def print_plan(plan: RhetoricPlan) -> None:
    """Pretty-print a RhetoricPlan."""
    print(f"\n{'=' * 70}")
    print("RHETORIC PLAN")
    print(f"{'=' * 70}\n")
    print(plan.summary())

    if plan.analogy:
        print(f"\n{'─' * 40}")
        print("STRUCTURAL ANALOGY (via Mercury diffusion)")
        print(f"{'─' * 40}")
        print(f"  {plan.analogy.source_domain} → {plan.analogy.target_domain}")
        print(f"\n  \"{plan.analogy.analogy_text}\"")
        if plan.analogy.mapping:
            print("\n  Mapping:")
            for src, tgt in plan.analogy.mapping.items():
                print(f"    {src} ↔ {tgt}")

    print(f"\n{'─' * 40}")
    print(f"DELIVERY: {plan.strategy.value}")
    print(f"{'─' * 40}")
    print(f"  {plan.strategy_description}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="toulmin",
        description="Formal verification of informal discourse (standalone)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command")
    plan_cmd = sub.add_parser("plan", help="Run full rhetoric pipeline on an argument")
    plan_cmd.add_argument("intent", help="The argument to analyze")
    plan_cmd.add_argument("--no-bridge", action="store_true", help="Skip analogy bridge")
    plan_cmd.add_argument("--contradict", nargs="*", default=[], help="Known contradicting evidence")

    val_cmd = sub.add_parser("validate", help="Validate a Toulmin graph from JSON file")
    val_cmd.add_argument("file", help="Path to JSON file containing an ArgumentGraph")

    sub.add_parser("demo", help="Run demo with built-in test cases (no API needed)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "plan":
        asyncio.run(run_plan(args))
    elif args.command == "validate":
        run_validate(args)
    elif args.command == "demo":
        run_demo()
    else:
        parser.print_help()
        sys.exit(1)

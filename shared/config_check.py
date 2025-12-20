"""Configuration check for skill activation.

Verifies skill configuration before activation and provides helpful
guidance when setup is incomplete.

Usage:
    from shared.config_check import check_skill_config, require_skill_config

    # Check without blocking
    status = check_skill_config("encyclopedia")
    if not status.ready:
        print(status.message)

    # Check and exit if not ready (for CLI entry points)
    require_skill_config("encyclopedia")  # Exits with code 78 if not configured
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

# Exit code for "configuration error" (BSD sysexits.h EX_CONFIG)
EX_CONFIG = 78

# Project root (cognitive-construct/)
PROJECT_ROOT = Path(__file__).parent.parent

# Skill configuration requirements
# Maps skill name -> (required_vars, recommended_vars, required_groups)
SKILL_REQUIREMENTS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "encyclopedia": (
        [],  # No strictly required vars
        ["EXA_API_KEY", "PERPLEXITY_API_KEY"],  # Recommended
        ["search"],  # Need at least one search provider
    ),
    "inland-empire": (
        [],  # Works with local SQLite
        ["MEM0_API_KEY"],
        [],
    ),
    "rhetoric": (
        [],
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"],
        ["llm"],  # Need at least 2 LLM providers
    ),
    "volition": (
        [],
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        ["llm"],  # Need at least 1 LLM provider
    ),
}

# Variable -> group mapping
VAR_GROUPS: dict[str, str] = {
    # Search providers
    "EXA_API_KEY": "search",
    "PERPLEXITY_API_KEY": "search",
    "CONTEXT7_API_KEY": "search",
    "KAGI_API_KEY": "search",
    "TAVILY_API_KEY": "search",
    "BRAVE_API_KEY": "search",
    "SERPER_API_KEY": "search",
    # LLM providers
    "OPENAI_API_KEY": "llm",
    "ANTHROPIC_API_KEY": "llm",
    "OPENROUTER_API_KEY": "llm",
    "GOOGLE_CLOUD_API_KEY": "llm",
    "MISTRAL_API_KEY": "llm",
    "XAI_API_KEY": "llm",
    "TOGETHER_API_KEY": "llm",
    "DEEPSEEK_API_KEY": "llm",
}

# Minimum LLM providers needed per skill
MIN_LLM_PROVIDERS: dict[str, int] = {
    "rhetoric": 2,
    "volition": 1,
}


@dataclass
class ConfigStatus:
    """Configuration status for a skill."""

    skill: str
    ready: bool
    configured: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    message: str = ""
    setup_command: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "skill": self.skill,
            "ready": self.ready,
            "configured": self.configured,
            "missing": self.missing,
            "message": self.message,
            "setup_command": self.setup_command,
        }


@lru_cache(maxsize=16)
def load_env_file(path: Path) -> dict[str, str]:
    """Load environment variables from a .env file (cached)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value:
                        env[key] = value
    except OSError:
        pass
    return env


def get_env_value(var_name: str, skill_name: str) -> str | None:
    """Get environment variable value from various sources."""
    # 1. Check current environment
    if var_name in os.environ:
        return os.environ[var_name]

    # 2. Check skill-specific .env.local
    skill_env = PROJECT_ROOT / skill_name / ".env.local"
    env = load_env_file(skill_env)
    if var_name in env:
        return env[var_name]

    # 3. Check project root .env.local
    root_env = PROJECT_ROOT / ".env.local"
    env = load_env_file(root_env)
    if var_name in env:
        return env[var_name]

    return None


def check_skill_config(skill_name: str) -> ConfigStatus:
    """Check if a skill is properly configured.

    Parameters
    ----------
    skill_name : str
        Name of the skill to check (e.g., "encyclopedia", "rhetoric").

    Returns
    -------
    ConfigStatus
        Status object with ready flag, configured vars, and guidance.
    """
    if skill_name not in SKILL_REQUIREMENTS:
        return ConfigStatus(
            skill=skill_name,
            ready=False,
            message=f"Unknown skill: {skill_name}",
        )

    required_vars, recommended_vars, required_groups = SKILL_REQUIREMENTS[skill_name]

    configured: list[str] = []
    missing: list[str] = []
    group_coverage: dict[str, int] = {}

    # Check all relevant variables
    all_vars = set(required_vars) | set(recommended_vars)

    # Also add all vars for required groups
    for var, group in VAR_GROUPS.items():
        if group in required_groups:
            all_vars.add(var)

    for var_name in all_vars:
        value = get_env_value(var_name, skill_name)
        if value:
            configured.append(var_name)
            group = VAR_GROUPS.get(var_name)
            if group:
                group_coverage[group] = group_coverage.get(group, 0) + 1
        elif var_name in required_vars:
            missing.append(var_name)

    # Check minimum LLM providers (more specific than generic group check)
    min_llm = MIN_LLM_PROVIDERS.get(skill_name, 0)
    if min_llm > 0:
        llm_count = group_coverage.get("llm", 0)
        if llm_count < min_llm:
            missing.append(
                f"(need at least {min_llm} LLM provider(s), have {llm_count})"
            )

    # Check required groups (skip llm if MIN_LLM_PROVIDERS handles it)
    for group in required_groups:
        if group not in group_coverage:
            # Skip generic message if MIN_LLM_PROVIDERS provides more specific check
            if group == "llm" and min_llm > 0:
                continue
            missing.append(f"(need at least one {group} provider)")

    is_ready = len(missing) == 0
    setup_cmd = f"uv run scripts/setup.py --get-keys {skill_name}"

    if is_ready:
        message = f"{skill_name} is ready ({len(configured)} provider(s) configured)"
    else:
        message = (
            f"{skill_name} needs configuration.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Run: {setup_cmd}"
        )

    return ConfigStatus(
        skill=skill_name,
        ready=is_ready,
        configured=configured,
        missing=missing,
        message=message,
        setup_command=setup_cmd,
    )


def require_skill_config(
    skill_name: str,
    exit_on_error: bool = True,
    output_format: Literal["text", "json"] = "text",
) -> ConfigStatus:
    """Check skill config and optionally exit if not ready.

    For use at the start of skill main() functions.

    Parameters
    ----------
    skill_name : str
        Name of the skill to check.
    exit_on_error : bool
        If True, exit with EX_CONFIG (78) when not configured.
    output_format : str
        Output format for error message ("text" or "json").

    Returns
    -------
    ConfigStatus
        Status object (only returned if ready or exit_on_error=False).
    """
    status = check_skill_config(skill_name)

    if status.ready:
        return status

    if output_format == "json":
        import json
        print(json.dumps({
            "error": "configuration_required",
            "skill": skill_name,
            "missing": status.missing,
            "configured": status.configured,
            "setup_command": status.setup_command,
            "message": f"Run '{status.setup_command}' to configure API keys",
        }), file=sys.stderr)
    else:
        # Rich output if available, plain text otherwise
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console(stderr=True)
            console.print()
            console.print(Panel(
                f"[bold yellow]{skill_name.title()}[/] needs configuration\n\n"
                f"[dim]Missing:[/] {', '.join(status.missing)}\n\n"
                f"[bold]To configure, run:[/]\n"
                f"  [cyan]{status.setup_command}[/]\n\n"
                f"[dim]Or for interactive setup:[/]\n"
                f"  [cyan]uv run scripts/setup.py[/]",
                title="Configuration Required",
                border_style="yellow",
            ))
            console.print()
        except ImportError:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f" {skill_name.title()} needs configuration", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"\nMissing: {', '.join(status.missing)}", file=sys.stderr)
            print(f"\nTo configure, run:", file=sys.stderr)
            print(f"  {status.setup_command}", file=sys.stderr)
            print(f"\nOr for interactive setup:", file=sys.stderr)
            print(f"  uv run scripts/setup.py\n", file=sys.stderr)

    if exit_on_error:
        sys.exit(EX_CONFIG)

    return status


def get_all_skills_status() -> dict[str, ConfigStatus]:
    """Get configuration status for all skills."""
    return {
        skill: check_skill_config(skill)
        for skill in SKILL_REQUIREMENTS
    }

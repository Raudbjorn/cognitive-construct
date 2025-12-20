#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich>=13.0", "httpx>=0.27"]
# ///
"""
Cognitive Construct Configuration Setup

Beautiful TUI for configuring all four skills with auto-detection and API testing.

Usage:
    uv run scripts/setup.py              # First-run wizard / interactive setup
    uv run scripts/setup.py --status     # Show configuration status
    uv run scripts/setup.py --test       # Test API connectivity
    uv run scripts/setup.py --export     # Export as shell commands
    uv run scripts/setup.py --get-keys   # Interactive API signup helper
    uv run scripts/setup.py --get-keys "1,3,encyclopedia"  # Open specific signups
"""

from __future__ import annotations

import os
import re
import sys
import time
import asyncio
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich.box import ROUNDED, DOUBLE
from rich import print as rprint

import httpx

# Constants
CONSTRUCT_ROOT = Path(__file__).parent.parent
SKILLS = ["encyclopedia", "inland-empire", "rhetoric", "volition"]
BROWSER_TAB_DELAY_SECONDS = 0.3  # Delay between opening browser tabs to prevent chaos
console = Console()

# Skill icons
SKILL_ICONS = {
    "encyclopedia": "📚",
    "inland-empire": "🧠",
    "rhetoric": "💭",
    "volition": "⚡",
}


class Requirement(Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass
class ConfigVar:
    """Configuration variable definition."""
    name: str
    description: str
    requirement: Requirement
    url: str = ""
    default: str = ""
    validator: Callable[[str], tuple[bool, str]] | None = None
    secret: bool = True
    group: str = ""
    test_endpoint: str = ""  # For connectivity testing
    benefit: str = ""  # What this API unlocks
    free_tier: str = ""  # Free tier info (empty = no free tier)


@dataclass
class SkillConfig:
    """Skill configuration definition."""
    name: str
    display_name: str
    description: str
    icon: str = ""
    variables: list[ConfigVar] = field(default_factory=list)
    min_required: int = 0
    required_groups: list[str] = field(default_factory=list)


# ============================================================================
# Validators
# ============================================================================

def validate_api_key_format(prefix: str) -> Callable[[str], tuple[bool, str]]:
    """Create validator for API key with expected prefix."""
    def validator(value: str) -> tuple[bool, str]:
        if not value:
            return False, "Value cannot be empty"
        if prefix and not value.startswith(prefix):
            return False, f"Expected key starting with '{prefix}'"
        if len(value) < 10:
            return False, "Key seems too short"
        return True, "Valid format"
    return validator


def validate_url(value: str) -> tuple[bool, str]:
    if not value:
        return False, "Value cannot be empty"
    if not re.match(r'^https?://', value):
        return False, "URL must start with http:// or https://"
    return True, "Valid URL"


def validate_nonempty(value: str) -> tuple[bool, str]:
    if not value or not value.strip():
        return False, "Value cannot be empty"
    return True, "Valid"


# ============================================================================
# Skill Configurations
# ============================================================================

ENCYCLOPEDIA_CONFIG = SkillConfig(
    name="encyclopedia",
    display_name="Encyclopedia",
    description="Knowledge retrieval from multiple sources",
    icon="📚",
    required_groups=["search"],
    variables=[
        ConfigVar(
            name="EXA_API_KEY",
            description="Exa AI - Web and code search",
            requirement=Requirement.RECOMMENDED,
            url="https://exa.ai/",
            validator=validate_nonempty,
            group="search",
            test_endpoint="exa",
            benefit="Neural web search, code search, research papers. Best for technical queries.",
            free_tier="1000 searches/month free",
        ),
        ConfigVar(
            name="PERPLEXITY_API_KEY",
            description="Perplexity AI - Research assistant",
            requirement=Requirement.RECOMMENDED,
            url="https://www.perplexity.ai/settings/api",
            validator=validate_nonempty,
            group="search",
            test_endpoint="perplexity",
            benefit="AI-powered research with citations. Great for current events and synthesis.",
            free_tier="Limited free tier",
        ),
        ConfigVar(
            name="CONTEXT7_API_KEY",
            description="Context7 - Library docs (works without key)",
            requirement=Requirement.OPTIONAL,
            url="https://context7.com/",
            benefit="Up-to-date library documentation. Works WITHOUT key (rate limited).",
            free_tier="Works without key!",
        ),
        ConfigVar(
            name="KAGI_API_KEY",
            description="Kagi Search - Premium search",
            requirement=Requirement.OPTIONAL,
            url="https://kagi.com/settings?p=api",
            validator=validate_nonempty,
            test_endpoint="kagi",
            benefit="Ad-free, privacy-focused search. High quality results.",
            free_tier="",  # No free tier
        ),
        ConfigVar(
            name="TAVILY_API_KEY",
            description="Tavily - AI search API",
            requirement=Requirement.OPTIONAL,
            url="https://tavily.com/",
            validator=validate_nonempty,
            test_endpoint="tavily",
            benefit="Search optimized for LLMs. Good for fact retrieval.",
            free_tier="1000 searches/month free",
        ),
        ConfigVar(
            name="BRAVE_API_KEY",
            description="Brave Search API",
            requirement=Requirement.OPTIONAL,
            url="https://brave.com/search/api/",
            validator=validate_nonempty,
            test_endpoint="brave",
            benefit="Independent search index. Privacy-focused.",
            free_tier="2000 queries/month free",
        ),
        ConfigVar(
            name="SERPER_API_KEY",
            description="Serper - Google Search API",
            requirement=Requirement.OPTIONAL,
            url="https://serper.dev/",
            validator=validate_nonempty,
            test_endpoint="serper",
            benefit="Google search results via API. Most comprehensive web coverage.",
            free_tier="2500 searches free (one-time)",
        ),
    ],
)

INLAND_EMPIRE_CONFIG = SkillConfig(
    name="inland-empire",
    display_name="Inland Empire",
    description="Memory substrate (works with local storage)",
    icon="🧠",
    variables=[
        ConfigVar(
            name="MEM0_API_KEY",
            description="Mem0 API for pattern memory",
            requirement=Requirement.OPTIONAL,
            url="https://app.mem0.ai/",
            validator=validate_nonempty,
            test_endpoint="mem0",
            benefit="Semantic memory with automatic extraction. Remembers patterns across sessions.",
            free_tier="1000 memories free",
        ),
        ConfigVar(
            name="LIBSQL_URL",
            description="LibSQL/Turso URL for fact memory",
            requirement=Requirement.OPTIONAL,
            url="https://turso.tech/",
            default="file:~/.inland-empire/construct.db",
            validator=validate_nonempty,
            secret=False,
            benefit="Graph-based fact storage. Entity relationships and structured knowledge.",
            free_tier="500 DBs, 9GB storage free (or use local SQLite)",
        ),
        ConfigVar(
            name="LIBSQL_AUTH_TOKEN",
            description="LibSQL/Turso auth token",
            requirement=Requirement.OPTIONAL,
            validator=validate_nonempty,
            benefit="Required for cloud Turso. Not needed for local SQLite.",
            free_tier="Included with Turso free tier",
        ),
        ConfigVar(
            name="QDRANT_API_KEY",
            description="Qdrant vector database",
            requirement=Requirement.OPTIONAL,
            url="https://qdrant.tech/",
            validator=validate_nonempty,
            benefit="Vector similarity search. Find semantically related memories.",
            free_tier="1GB free cluster",
        ),
    ],
)

RHETORIC_CONFIG = SkillConfig(
    name="rhetoric",
    display_name="Rhetoric",
    description="Reasoning engine (needs 2+ LLM keys)",
    icon="💭",
    min_required=2,
    required_groups=["llm"],
    variables=[
        ConfigVar(
            name="OPENAI_API_KEY",
            description="OpenAI API",
            requirement=Requirement.RECOMMENDED,
            url="https://platform.openai.com/api-keys",
            validator=validate_api_key_format("sk-"),
            group="llm",
            test_endpoint="openai",
            benefit="GPT-4o, o1 reasoning. Best for complex analysis and code.",
            free_tier="$5 free credit (new accounts)",
        ),
        ConfigVar(
            name="ANTHROPIC_API_KEY",
            description="Anthropic Claude API",
            requirement=Requirement.RECOMMENDED,
            url="https://console.anthropic.com/settings/keys",
            validator=validate_api_key_format("sk-ant-"),
            group="llm",
            test_endpoint="anthropic",
            benefit="Claude Opus/Sonnet. Excellent for nuanced reasoning and safety.",
            free_tier="$5 free credit (new accounts)",
        ),
        ConfigVar(
            name="OPENROUTER_API_KEY",
            description="OpenRouter (access to many models)",
            requirement=Requirement.RECOMMENDED,
            url="https://openrouter.ai/keys",
            validator=validate_api_key_format("sk-or-"),
            group="llm",
            test_endpoint="openrouter",
            benefit="Access 100+ models via one API. Good for model diversity in deliberation.",
            free_tier="Some free models available",
        ),
        ConfigVar(
            name="GOOGLE_CLOUD_API_KEY",
            description="Google Cloud / Gemini API",
            requirement=Requirement.OPTIONAL,
            url="https://aistudio.google.com/apikey",
            validator=validate_api_key_format("AIza"),
            group="llm",
            test_endpoint="google",
            benefit="Gemini 2.0 Flash/Pro. Fast, large context, multimodal.",
            free_tier="Generous free tier (15 RPM)",
        ),
        ConfigVar(
            name="MISTRAL_API_KEY",
            description="Mistral AI",
            requirement=Requirement.OPTIONAL,
            url="https://console.mistral.ai/api-keys",
            validator=validate_nonempty,
            group="llm",
            test_endpoint="mistral",
            benefit="Mixtral, Mistral Large. Strong open-weight models.",
            free_tier="Limited free tier",
        ),
        ConfigVar(
            name="XAI_API_KEY",
            description="xAI / Grok API",
            requirement=Requirement.OPTIONAL,
            url="https://console.x.ai/",
            validator=validate_nonempty,
            group="llm",
            test_endpoint="xai",
            benefit="Grok models. Real-time X/Twitter data access.",
            free_tier="$25 free credit/month",
        ),
        ConfigVar(
            name="TOGETHER_API_KEY",
            description="Together AI",
            requirement=Requirement.OPTIONAL,
            url="https://api.together.xyz/",
            validator=validate_nonempty,
            group="llm",
            test_endpoint="together",
            benefit="Fast inference for open models. Llama, Qwen, DeepSeek.",
            free_tier="$5 free credit",
        ),
        ConfigVar(
            name="DEFAULT_LLM_PROVIDER",
            description="Default provider (gemini/openai/anthropic)",
            requirement=Requirement.OPTIONAL,
            default="gemini",
            validator=validate_nonempty,
            secret=False,
            benefit="Which LLM to use by default when not specified.",
            free_tier="N/A (config only)",
        ),
    ],
)

VOLITION_CONFIG = SkillConfig(
    name="volition",
    display_name="Volition",
    description="Agency and execution (needs 1+ LLM key)",
    icon="⚡",
    min_required=1,
    required_groups=["llm"],
    variables=[
        ConfigVar(
            name="OPENAI_API_KEY",
            description="OpenAI API",
            requirement=Requirement.RECOMMENDED,
            url="https://platform.openai.com/api-keys",
            validator=validate_api_key_format("sk-"),
            group="llm",
            test_endpoint="openai",
            benefit="GPT-4o for code editing and web search integration.",
            free_tier="$5 free credit (new accounts)",
        ),
        ConfigVar(
            name="ANTHROPIC_API_KEY",
            description="Anthropic Claude API",
            requirement=Requirement.RECOMMENDED,
            url="https://console.anthropic.com/settings/keys",
            validator=validate_api_key_format("sk-ant-"),
            group="llm",
            test_endpoint="anthropic",
            benefit="Claude for careful code review and safe execution.",
            free_tier="$5 free credit (new accounts)",
        ),
        ConfigVar(
            name="DEEPSEEK_API_KEY",
            description="DeepSeek (code-specialized)",
            requirement=Requirement.OPTIONAL,
            url="https://platform.deepseek.com/",
            validator=validate_nonempty,
            group="llm",
            test_endpoint="deepseek",
            benefit="DeepSeek Coder. Excellent for code generation, very cheap.",
            free_tier="$5 free credit",
        ),
        ConfigVar(
            name="SHODAN_API_KEY",
            description="Shodan security reconnaissance",
            requirement=Requirement.OPTIONAL,
            url="https://account.shodan.io/",
            validator=validate_nonempty,
            test_endpoint="shodan",
            benefit="Security reconnaissance. Find exposed services, vulnerabilities.",
            free_tier="Limited free tier (account required)",
        ),
    ],
)

ALL_CONFIGS = [ENCYCLOPEDIA_CONFIG, INLAND_EMPIRE_CONFIG, RHETORIC_CONFIG, VOLITION_CONFIG]


# ============================================================================
# Environment Detection & Loading
# ============================================================================

def load_env_file(path: Path) -> dict[str, str]:
    """Load environment variables from a .env file."""
    env = {}
    if not path.exists():
        return env
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
    return env


def escape_env_value(value: str) -> str:
    """Escape special characters in env value for .env file format."""
    # Escape backslashes first, then quotes, then newlines
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    return value


def save_env_file(path: Path, env: dict[str, str], skill_name: str):
    """Save environment variables to a .env file with nice formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    header = "# {} Skill Configuration".format(skill_name.replace("-", " ").title())
    lines = [
        header,
        "# Generated by Cognitive Construct Setup",
        "",
    ]

    for key, value in sorted(env.items()):
        escaped = escape_env_value(value)
        lines.append('{}="{}"'.format(key, escaped))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_skill_env_path(skill_name: str) -> Path:
    return CONSTRUCT_ROOT / skill_name / ".env.local"


def load_all_config() -> dict[str, dict[str, str]]:
    """Load configuration from all sources."""
    config = {skill: {} for skill in SKILLS}

    # Load from skill .env.local files
    for skill in SKILLS:
        env_path = get_skill_env_path(skill)
        config[skill] = load_env_file(env_path)

    return config


def auto_detect_env_vars() -> dict[str, str]:
    """Auto-detect environment variables from various sources."""
    detected = {}

    # 1. Current environment
    all_var_names = set()
    for skill_config in ALL_CONFIGS:
        for var in skill_config.variables:
            all_var_names.add(var.name)

    for name in all_var_names:
        if name in os.environ:
            detected[name] = os.environ[name]

    # 2. Common .env file locations
    search_paths = [
        Path.home() / ".env",
        Path.home() / ".env.local",
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
        CONSTRUCT_ROOT / ".env",
        CONSTRUCT_ROOT / ".env.local",
    ]

    for path in search_paths:
        if path.exists():
            env = load_env_file(path)
            for name in all_var_names:
                if name in env and name not in detected:
                    detected[name] = env[name]

    return detected


def get_current_value(var_name: str, all_config: dict[str, dict[str, str]], detected: dict[str, str] | None = None) -> str | None:
    """Get current value for a variable from all sources."""
    # Check detected first
    if detected and var_name in detected:
        return detected[var_name]

    # Check environment
    if var_name in os.environ:
        return os.environ[var_name]

    # Check skill configs
    for skill_config in all_config.values():
        if var_name in skill_config:
            return skill_config[var_name]

    return None


# ============================================================================
# API Connectivity Testing
# ============================================================================

def get_api_test_config(name: str, key: str) -> tuple[str, dict[str, str], str, dict | None] | None:
    """Get test configuration for an API. Returns (url, headers, method, body) or None."""
    tests = {
        "openai": ("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        "anthropic": ("https://api.anthropic.com/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01"}, "GET", None),
        "openrouter": ("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        "google": (f"https://generativelanguage.googleapis.com/v1/models?key={key}", {}, "GET", None),
        "mistral": ("https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        "deepseek": ("https://api.deepseek.com/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        "together": ("https://api.together.xyz/v1/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        "xai": ("https://api.x.ai/v1/models", {"Authorization": f"Bearer {key}"}, "GET", None),
        # Note: Exa API requires camelCase "numResults" per their API spec
        "exa": ("https://api.exa.ai/search", {"x-api-key": key, "Content-Type": "application/json"}, "POST", {"query": "test", "numResults": 1}),
        "perplexity": ("https://api.perplexity.ai/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, "POST", {"model": "sonar", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}),
        "kagi": ("https://kagi.com/api/v0/enrich/web?q=test", {"Authorization": f"Bot {key}"}, "GET", None),
        "tavily": ("https://api.tavily.com/search", {"Content-Type": "application/json"}, "POST", {"api_key": key, "query": "test", "max_results": 1}),
        "brave": ("https://api.search.brave.com/res/v1/web/search?q=test", {"X-Subscription-Token": key}, "GET", None),
        "serper": ("https://google.serper.dev/search", {"X-API-KEY": key, "Content-Type": "application/json"}, "POST", {"q": "test"}),
        "mem0": ("https://api.mem0.ai/v1/memories/?user_id=test", {"Authorization": f"Token {key}"}, "GET", None),
        "shodan": (f"https://api.shodan.io/api-info?key={key}", {}, "GET", None),
    }
    return tests.get(name)


async def test_api(name: str, key: str, client: httpx.AsyncClient) -> tuple[str, bool, str]:
    """Test API connectivity using shared client. Returns (name, success, message)."""
    config = get_api_test_config(name, key)
    if config is None:
        return name, False, "No test available"

    url, headers, method, body = config

    try:
        if method == "GET":
            resp = await client.get(url, headers=headers)
        else:
            resp = await client.post(url, headers=headers, json=body)

        if resp.status_code in (200, 201):
            return name, True, "Connected"
        elif resp.status_code == 401:
            return name, False, "Invalid key"
        elif resp.status_code == 403:
            # Check for xAI-specific "no credits" error
            if name == "xai":
                try:
                    data = resp.json()
                    error_msg = data.get("error")
                    if isinstance(error_msg, str) and "credits" in error_msg.lower():
                        return name, True, "No credits"
                except ValueError:
                    pass  # JSON parsing failed
            return name, False, "Access denied"
        elif resp.status_code == 429:
            return name, True, "Rate limited (key valid)"
        else:
            # Check for Kagi-specific "insufficient credit" error
            if name == "kagi" and resp.status_code == 400:
                try:
                    data = resp.json()
                    errors = data.get("error")
                    if isinstance(errors, list) and any(
                        isinstance(e, dict) and e.get("code") == 101 for e in errors
                    ):
                        return name, True, "No credits"
                except ValueError:
                    pass  # JSON parsing failed
            return name, False, "HTTP {}".format(resp.status_code)
    except httpx.TimeoutException:
        return name, False, "Timeout"
    except httpx.RequestError:
        return name, False, "Network error"
    except Exception:
        return name, False, "Unexpected error"


async def run_all_tests(all_config: dict[str, dict[str, str]], detected: dict[str, str]) -> list[tuple[str, str, bool, str]]:
    """Run all API tests concurrently using a shared client."""
    var_to_test: dict[str, tuple[str, str]] = {}

    # Collect all variables that have test endpoints and values
    for skill_config in ALL_CONFIGS:
        for var in skill_config.variables:
            if var.test_endpoint and var.name not in var_to_test:
                value = get_current_value(var.name, all_config, detected)
                if value:
                    var_to_test[var.name] = (var.test_endpoint, value)

    # Run all tests concurrently with shared client
    async with httpx.AsyncClient(timeout=10) as client:
        names = list(var_to_test.keys())
        coros = [test_api(endpoint, key, client) for endpoint, key in var_to_test.values()]
        results_raw = await asyncio.gather(*coros, return_exceptions=True)

    # Normalize results, handling unexpected exceptions
    results: list[tuple[str, str, bool, str]] = []
    for var_name, res in zip(names, results_raw):
        if isinstance(res, Exception):
            results.append((var_name, "unknown", False, str(res)[:30]))
        else:
            endpoint, success, msg = res
            results.append((var_name, endpoint, success, msg))

    return results


# ============================================================================
# Status Display
# ============================================================================

def check_skill_status(skill_config: SkillConfig, all_config: dict[str, dict[str, str]], detected: dict[str, str] | None = None) -> tuple[bool, list[str], list[str]]:
    """Check if a skill is properly configured."""
    configured = []
    missing = []
    group_coverage = {}

    for var in skill_config.variables:
        value = get_current_value(var.name, all_config, detected)
        if value:
            configured.append(var.name)
            if var.group:
                group_coverage[var.group] = True
        elif var.requirement == Requirement.REQUIRED:
            missing.append(var.name)

    for group in skill_config.required_groups:
        if group not in group_coverage:
            for var in skill_config.variables:
                if var.group == group:
                    if var.name not in missing:
                        missing.append(f"(one of {group} group)")
                    break

    is_ready = len(missing) == 0
    if skill_config.min_required > 0:
        llm_count = sum(1 for var in skill_config.variables
                       if var.group == "llm" and get_current_value(var.name, all_config, detected))
        if llm_count < skill_config.min_required:
            is_ready = False
            if not any("llm" in m for m in missing):
                missing.append(f"(need {skill_config.min_required}+ LLM keys)")

    return is_ready, configured, missing


def create_status_table(all_config: dict[str, dict[str, str]], detected: dict[str, str] | None = None) -> Table:
    """Create a rich table showing configuration status."""
    table = Table(title="🧠 Cognitive Construct Status", box=ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Skill", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Configured", style="green")
    table.add_column("Missing", style="yellow")

    for skill_config in ALL_CONFIGS:
        is_ready, configured, missing = check_skill_status(skill_config, all_config, detected)

        status = "[green]✓ Ready[/]" if is_ready else "[yellow]○ Setup needed[/]"
        icon = SKILL_ICONS.get(skill_config.name, "")

        configured_str = ", ".join(configured[:3])
        if len(configured) > 3:
            configured_str += f" +{len(configured) - 3}"

        missing_str = ", ".join(missing[:2])
        if len(missing) > 2:
            missing_str += f" +{len(missing) - 2}"

        table.add_row(
            f"{icon} {skill_config.display_name}",
            status,
            configured_str or "-",
            missing_str or "-",
        )

    return table


def show_status():
    """Display current configuration status."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Cognitive Construct[/] Configuration Status",
        border_style="cyan",
    ))

    all_config = load_all_config()
    detected = auto_detect_env_vars()

    if detected:
        console.print(f"\n[dim]Auto-detected {len(detected)} environment variable(s)[/]")

    console.print()
    console.print(create_status_table(all_config, detected))
    console.print()


# ============================================================================
# Connectivity Test Display
# ============================================================================

def show_test_results():
    """Test and display API connectivity."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔌 API Connectivity Test[/]",
        border_style="cyan",
    ))

    all_config = load_all_config()
    detected = auto_detect_env_vars()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Testing APIs...", total=100)

        try:
            results = asyncio.run(run_all_tests(all_config, detected))
        except RuntimeError as e:
            if "running event loop" in str(e).lower():
                console.print("\n[red]Error: Cannot run in existing event loop (IDE/notebook).[/]")
                console.print("[dim]Run from terminal: uv run scripts/setup.py --test[/]")
                return
            raise
        progress.update(task, completed=100)

    if not results:
        console.print("\n[yellow]No API keys configured to test[/]")
        return

    table = Table(title="API Test Results", box=ROUNDED)
    table.add_column("Variable", style="bold")
    table.add_column("API", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Message")

    for var_name, endpoint, success, msg in sorted(results):
        status = "[green]✓[/]" if success else "[red]✗[/]"
        msg_text = msg if success else f"[red]{msg}[/]"
        table.add_row(var_name, endpoint, status, msg_text)

    console.print()
    console.print(table)

    success_count = sum(1 for _, _, s, _ in results if s)
    console.print(f"\n[{'green' if success_count == len(results) else 'yellow'}]{success_count}/{len(results)} APIs connected successfully[/]")


# ============================================================================
# Interactive Setup
# ============================================================================

def mask_secret(value: str) -> str:
    """Mask a secret value for display."""
    if len(value) <= 8:
        return "****"
    return value[:4] + "..." + value[-4:]


def configure_skill_interactive(skill_config: SkillConfig, all_config: dict[str, dict[str, str]], detected: dict[str, str]):
    """Interactively configure a single skill."""
    console.print()
    console.print(Panel(
        f"[bold]{skill_config.icon} {skill_config.display_name}[/]\n{skill_config.description}",
        border_style="cyan",
    ))

    env_path = get_skill_env_path(skill_config.name)
    current_env = dict(all_config.get(skill_config.name, {}))
    new_env = dict(current_env)

    for var in skill_config.variables:
        current = get_current_value(var.name, all_config, detected)

        # Build prompt
        req_badge = {
            Requirement.REQUIRED: "[red](required)[/]",
            Requirement.RECOMMENDED: "[yellow](recommended)[/]",
            Requirement.OPTIONAL: "[dim](optional)[/]",
        }[var.requirement]

        console.print(f"\n[bold]{var.name}[/] {req_badge}")
        console.print(f"  [dim]{var.description}[/]")
        if var.url:
            console.print(f"  [blue link={var.url}]Get key →[/]")

        if current:
            display = mask_secret(current) if var.secret else current
            console.print(f"  [green]Current: {display}[/]")

        default_hint = f" [{var.default}]" if var.default else ""

        # Inner loop for retry on validation failure
        while True:
            try:
                if var.secret:
                    value = Prompt.ask(f"  Enter value{default_hint}", password=True, default="")
                else:
                    value = Prompt.ask(f"  Enter value{default_hint}", default=var.default or "")
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled[/]")
                return

            value = value.strip()

            if not value:
                if var.default:
                    value = var.default
                else:
                    break  # Skip this variable

            # Validate
            if var.validator:
                is_valid, msg = var.validator(value)
                if not is_valid:
                    console.print(f"  [red]✗ {msg}[/]")
                    if Confirm.ask("  Try again?", default=True):
                        continue  # Re-prompt for same variable
                    break  # Skip this variable
                console.print(f"  [green]✓ {msg}[/]")

            new_env[var.name] = value
            break  # Successfully validated, move to next variable

    # Save
    if new_env != current_env:
        save_env_file(env_path, new_env, skill_config.name)
        console.print(f"\n[green]✓ Saved to {env_path}[/]")
        all_config[skill_config.name] = new_env
    else:
        console.print("\n[dim]No changes made[/]")


def first_run_wizard():
    """First-run wizard with auto-detection."""
    console.clear()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🧠 Cognitive Construct Setup Wizard[/]\n\n"
        "This wizard will help you configure the four cognitive skills:\n"
        "  📚 Encyclopedia - Knowledge retrieval\n"
        "  🧠 Inland Empire - Persistent memory\n"
        "  💭 Rhetoric - Structured reasoning\n"
        "  ⚡ Volition - Agency and execution",
        border_style="cyan",
        box=DOUBLE,
    ))

    # Auto-detect
    console.print("\n[bold]Scanning for existing configuration...[/]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Detecting environment variables...", total=None)
        detected = auto_detect_env_vars()
        all_config = load_all_config()
        if sys.stdin.isatty():
            time.sleep(0.5)  # Brief pause for effect in interactive mode

    if detected:
        console.print(f"\n[green]✓ Found {len(detected)} existing API key(s)![/]")

        table = Table(box=ROUNDED, show_header=False)
        table.add_column("Variable", style="cyan")
        table.add_column("Value", style="green")

        for name, value in sorted(detected.items()):
            table.add_row(name, mask_secret(value))

        console.print(table)

        if Confirm.ask("\n[bold]Import these detected keys?[/]", default=True):
            # Distribute to appropriate skills (only save if changed)
            for skill_config in ALL_CONFIGS:
                original_env = all_config.get(skill_config.name, {})
                skill_env = dict(original_env)
                for var in skill_config.variables:
                    if var.name in detected and var.name not in skill_env:
                        skill_env[var.name] = detected[var.name]
                if skill_env != original_env:
                    save_env_file(get_skill_env_path(skill_config.name), skill_env, skill_config.name)
                    all_config[skill_config.name] = skill_env
            console.print("[green]✓ Keys imported![/]")
            all_config = load_all_config()  # Reload
    else:
        console.print("\n[yellow]No existing API keys detected[/]")

    # Show current status
    console.print()
    console.print(create_status_table(all_config, detected))

    # Ask what to configure
    console.print()
    skills_needing_setup = []
    for skill_config in ALL_CONFIGS:
        is_ready, _, _ = check_skill_status(skill_config, all_config, detected)
        if not is_ready:
            skills_needing_setup.append(skill_config)

    if not skills_needing_setup:
        console.print("[green]✓ All skills are configured![/]")
        if Confirm.ask("\nWould you like to test API connectivity?", default=True):
            show_test_results()
        return

    console.print(f"[yellow]{len(skills_needing_setup)} skill(s) need configuration[/]")

    if Confirm.ask("\nConfigure missing skills now?", default=True):
        for skill_config in skills_needing_setup:
            configure_skill_interactive(skill_config, all_config, detected)
            all_config = load_all_config()  # Reload after each

    # Final status
    console.print()
    console.print(create_status_table(all_config, detected))

    if Confirm.ask("\nTest API connectivity?", default=True):
        show_test_results()

    console.print("\n[bold green]Setup complete![/] 🎉")
    console.print("[dim]Run 'uv run scripts/setup.py --status' anytime to check configuration[/]")


def interactive_menu():
    """Main interactive menu."""
    all_config = load_all_config()
    detected = auto_detect_env_vars()

    while True:
        console.clear()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🧠 Cognitive Construct[/] Configuration",
            border_style="cyan",
        ))
        console.print()
        console.print(create_status_table(all_config, detected))

        console.print("\n[bold]What would you like to do?[/]\n")
        console.print("  [cyan]1[/] Configure Encyclopedia (knowledge)")
        console.print("  [cyan]2[/] Configure Inland Empire (memory)")
        console.print("  [cyan]3[/] Configure Rhetoric (reasoning)")
        console.print("  [cyan]4[/] Configure Volition (agency)")
        console.print("  [cyan]5[/] Configure all skills")
        console.print("  [cyan]6[/] Test API connectivity")
        console.print("  [cyan]7[/] Export as shell commands")
        console.print("  [cyan]8[/] Exit")

        choice = Prompt.ask("\n[cyan]Enter choice[/]", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="8")

        if choice == "1":
            configure_skill_interactive(ENCYCLOPEDIA_CONFIG, all_config, detected)
        elif choice == "2":
            configure_skill_interactive(INLAND_EMPIRE_CONFIG, all_config, detected)
        elif choice == "3":
            configure_skill_interactive(RHETORIC_CONFIG, all_config, detected)
        elif choice == "4":
            configure_skill_interactive(VOLITION_CONFIG, all_config, detected)
        elif choice == "5":
            for config in ALL_CONFIGS:
                configure_skill_interactive(config, all_config, detected)
                all_config = load_all_config()
        elif choice == "6":
            show_test_results()
            Prompt.ask("\nPress Enter to continue")
        elif choice == "7":
            export_shell(all_config, detected)
            Prompt.ask("\nPress Enter to continue")
        elif choice == "8":
            break

        all_config = load_all_config()


def export_shell(all_config: dict[str, dict[str, str]] | None = None, detected: dict[str, str] | None = None):
    """Export configuration as shell commands."""
    if all_config is None:
        all_config = load_all_config()
    if detected is None:
        detected = auto_detect_env_vars()

    console.print("\n[bold]# Shell Export[/]")
    console.print("[dim]# Add to ~/.bashrc or ~/.zshrc, or run:[/]")
    console.print("[dim]# eval $(uv run scripts/setup.py --export 2>/dev/null | grep ^export)[/]\n")

    exported = set()
    for skill_config in ALL_CONFIGS:
        console.print(f"# {skill_config.display_name}")
        for var in skill_config.variables:
            if var.name in exported:
                continue
            value = get_current_value(var.name, all_config, detected)
            if value:
                escaped = value.replace("'", "'\"'\"'")
                console.print(f"export {var.name}='{escaped}'")
                exported.add(var.name)
        console.print()


# ============================================================================
# Get Keys - Interactive API Signup
# ============================================================================

@dataclass
class ApiEntry:
    """API entry for the signup list."""
    index: int
    skill: str
    var_name: str
    description: str
    url: str
    benefit: str
    free_tier: str
    requirement: Requirement


def build_api_list() -> list[ApiEntry]:
    """Build numbered list of all APIs with signup URLs."""
    entries: list[ApiEntry] = []
    seen_vars: set[str] = set()
    index = 1

    for skill_config in ALL_CONFIGS:
        for var in skill_config.variables:
            # Skip vars without URLs or duplicates (shared across skills)
            if not var.url or var.name in seen_vars:
                continue
            seen_vars.add(var.name)

            entries.append(ApiEntry(
                index=index,
                skill=skill_config.name,
                var_name=var.name,
                description=var.description,
                url=var.url,
                benefit=var.benefit,
                free_tier=var.free_tier,
                requirement=var.requirement,
            ))
            index += 1

    return entries


def parse_selection(selection: str, entries: list[ApiEntry]) -> list[ApiEntry]:
    """Parse selection input like '1,3,5-7,encyclopedia' into list of entries.

    Supports:
    - Single numbers: "1", "3"
    - Ranges: "5-7" (inclusive)
    - Skill names: "encyclopedia", "rhetoric"
    - Combinations: "1,3,5-7,encyclopedia"
    - "all" for everything
    """
    if not selection.strip():
        return []

    if selection.strip().lower() == "all":
        return entries

    selected: list[ApiEntry] = []
    selected_indices: set[int] = set()
    # O(1) lookup by index for numeric selections
    index_to_entry = {entry.index: entry for entry in entries}

    for part in selection.split(","):
        part = part.strip().lower()
        if not part:
            continue

        # Check if it's a skill name
        skill_match = None
        for skill_name in SKILLS:
            if part == skill_name or part == skill_name.replace("-", ""):
                skill_match = skill_name
                break

        if skill_match:
            # Add all entries for this skill
            for entry in entries:
                if entry.skill == skill_match and entry.index not in selected_indices:
                    selected.append(entry)
                    selected_indices.add(entry.index)
            continue

        # Check if it's a range (e.g., "5-7")
        if "-" in part and part[0] != "-":
            range_parts = part.split("-")
            if len(range_parts) == 2:
                try:
                    start = int(range_parts[0])
                    end = int(range_parts[1])
                    for i in range(start, end + 1):
                        if i in index_to_entry and i not in selected_indices:
                            selected.append(index_to_entry[i])
                            selected_indices.add(i)
                    continue
                except ValueError:
                    pass

        # Try as single number
        try:
            num = int(part)
            if num in index_to_entry and num not in selected_indices:
                selected.append(index_to_entry[num])
                selected_indices.add(num)
        except ValueError:
            console.print(f"[yellow]Ignoring unrecognized: '{part}'[/]")

    return selected


def create_api_table(entries: list[ApiEntry], all_config: dict[str, dict[str, str]], detected: dict[str, str]) -> Table:
    """Create table showing all APIs with benefits."""
    table = Table(
        title="🔑 API Key Signup Guide",
        box=ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="bold cyan", width=3)
    table.add_column("Skill", style="dim", width=12)
    table.add_column("API", width=20)
    table.add_column("Benefit", width=40)
    table.add_column("Free Tier", width=20)
    table.add_column("Status", justify="center", width=8)

    current_skill = None
    for entry in entries:
        # Add separator between skills
        if current_skill is not None and entry.skill != current_skill:
            table.add_row("", "", "", "", "", "")
        current_skill = entry.skill

        # Check if already configured
        value = get_current_value(entry.var_name, all_config, detected)
        status = "[green]✓[/]" if value else "[dim]○[/]"

        # Requirement badge
        req_color = {"required": "red", "recommended": "yellow", "optional": "dim"}
        req = f"[{req_color[entry.requirement.value]}]{entry.requirement.value[0].upper()}[/]"

        # Free tier styling
        free_tier = entry.free_tier
        if free_tier:
            if "free" in free_tier.lower() or "without key" in free_tier.lower():
                free_tier = f"[green]{free_tier}[/]"
            else:
                free_tier = f"[dim]{free_tier}[/]"
        else:
            free_tier = "[red]Paid only[/]"

        icon = SKILL_ICONS.get(entry.skill, "")
        skill_display = f"{icon} {entry.skill}"

        table.add_row(
            str(entry.index),
            skill_display,
            f"{entry.description} {req}",
            entry.benefit or "[dim]-[/]",
            free_tier,
            status,
        )

    return table


def show_get_keys(selection: str | None = None):
    """Interactive API key signup helper."""
    all_config = load_all_config()
    detected = auto_detect_env_vars()
    entries = build_api_list()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔑 API Key Signup Helper[/]\n\n"
        "Select which APIs you'd like to sign up for.\n"
        "Browser tabs will open for each selection.",
        border_style="cyan",
    ))

    # Show the table
    console.print()
    console.print(create_api_table(entries, all_config, detected))

    # Legend
    console.print("\n[dim]Status: ✓ = configured, ○ = not configured[/]")
    console.print("[dim]Requirement: [red]R[/]=required [yellow]R[/]=recommended [dim]O[/]=optional[/]")

    # Get selection
    console.print("\n[bold]Enter selection:[/]")
    console.print("[dim]  Examples: 1,3,5-7  |  encyclopedia  |  all  |  1-5,rhetoric[/]")
    console.print("[dim]  Press Enter without input to cancel[/]")

    if selection is None:
        try:
            selection = Prompt.ask("\n[cyan]Selection[/]", default="")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/]")
            return

    if not selection.strip():
        console.print("[dim]No selection made[/]")
        return

    # Parse and open
    selected = parse_selection(selection, entries)

    if not selected:
        console.print("[yellow]No valid selections found[/]")
        return

    console.print(f"\n[bold]Opening {len(selected)} signup page(s)...[/]")

    opened_count = 0
    for entry in selected:
        console.print(f"  [cyan]→[/] {entry.description}: [blue link={entry.url}]{entry.url}[/]")
        try:
            webbrowser.open(entry.url)
            opened_count += 1
        except Exception:
            # Fallback for headless systems - URL is already printed above
            pass
        time.sleep(BROWSER_TAB_DELAY_SECONDS)

    if opened_count == len(selected):
        console.print(f"\n[green]✓ Opened {opened_count} page(s) in browser[/]")
    elif opened_count > 0:
        console.print(f"\n[yellow]Opened {opened_count}/{len(selected)} page(s) (some failed)[/]")
    else:
        console.print("\n[yellow]Could not open browser. URLs are printed above.[/]")
    console.print("[dim]After signing up, run 'uv run scripts/setup.py' to configure the keys[/]")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cognitive Construct Configuration Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/setup.py              # First-run wizard
  uv run scripts/setup.py --status     # Show status
  uv run scripts/setup.py --test       # Test API connectivity
  uv run scripts/setup.py --export     # Export as shell commands
  uv run scripts/setup.py --get-keys   # Interactive API signup helper
  uv run scripts/setup.py --get-keys "1,3,5-7,encyclopedia"  # Open specific signups
  uv run scripts/setup.py --menu       # Interactive menu
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show configuration status")
    parser.add_argument("--test", action="store_true", help="Test API connectivity")
    parser.add_argument("--export", action="store_true", help="Export as shell commands")
    parser.add_argument("--get-keys", nargs="?", const="", metavar="SELECTION",
                       help="Open API signup pages (e.g., '1,3,5-7,encyclopedia' or 'all')")
    parser.add_argument("--menu", action="store_true", help="Interactive menu")
    parser.add_argument("--wizard", action="store_true", help="Run first-run wizard")

    args = parser.parse_args()

    try:
        if args.status:
            show_status()
        elif args.test:
            show_test_results()
        elif args.export:
            export_shell()
        elif args.get_keys is not None:
            # --get-keys with or without selection argument
            show_get_keys(args.get_keys if args.get_keys else None)
        elif args.menu:
            interactive_menu()
        elif args.wizard:
            first_run_wizard()
        else:
            # Default: check if first run, then wizard or menu
            all_config = load_all_config()
            has_any_config = any(bool(c) for c in all_config.values())

            if has_any_config:
                interactive_menu()
            else:
                first_run_wizard()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()

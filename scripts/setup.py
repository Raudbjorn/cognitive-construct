#!/usr/bin/env python3
"""
Cognitive Construct Configuration Setup

Interactive tool for configuring all four skills:
- Encyclopedia (knowledge retrieval)
- Inland Empire (memory)
- Rhetoric (reasoning)
- Volition (agency)

Usage:
    python scripts/setup.py              # Interactive setup
    python scripts/setup.py --status     # Show configuration status
    python scripts/setup.py --export     # Export as shell commands
    python scripts/setup.py --validate   # Validate existing configuration
"""

from __future__ import annotations

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum

# Constants
CONSTRUCT_ROOT = Path(__file__).parent.parent
SKILLS = ["encyclopedia", "inland-empire", "rhetoric", "volition"]


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


@dataclass
class SkillConfig:
    """Skill configuration definition."""
    name: str
    display_name: str
    description: str
    variables: list[ConfigVar] = field(default_factory=list)
    min_required: int = 0  # Minimum number of required vars that must be set
    required_groups: list[str] = field(default_factory=list)  # At least one from each group


# Validators
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
    """Validate URL format."""
    if not value:
        return False, "Value cannot be empty"
    if not re.match(r'^https?://', value):
        return False, "URL must start with http:// or https://"
    return True, "Valid URL format"


def validate_postgres_url(value: str) -> tuple[bool, str]:
    """Validate PostgreSQL URL format."""
    if not value:
        return False, "Value cannot be empty"
    if not re.match(r'^postgresql://', value):
        return False, "URL must start with postgresql://"
    return True, "Valid PostgreSQL URL"


def validate_bolt_url(value: str) -> tuple[bool, str]:
    """Validate Neo4j Bolt URL format."""
    if not value:
        return False, "Value cannot be empty"
    if not re.match(r'^bolt://', value):
        return False, "URL must start with bolt://"
    return True, "Valid Bolt URL"


def validate_nonempty(value: str) -> tuple[bool, str]:
    """Validate non-empty value."""
    if not value or not value.strip():
        return False, "Value cannot be empty"
    return True, "Valid"


# Skill configurations
ENCYCLOPEDIA_CONFIG = SkillConfig(
    name="encyclopedia",
    display_name="Encyclopedia",
    description="Knowledge retrieval from multiple sources",
    required_groups=["search"],
    variables=[
        ConfigVar(
            name="EXA_API_KEY",
            description="Exa AI - Web and code search",
            requirement=Requirement.RECOMMENDED,
            url="https://exa.ai/",
            validator=validate_nonempty,
            group="search",
        ),
        ConfigVar(
            name="PERPLEXITY_API_KEY",
            description="Perplexity AI - Research assistant",
            requirement=Requirement.RECOMMENDED,
            url="https://www.perplexity.ai/settings/api",
            validator=validate_nonempty,
            group="search",
        ),
        ConfigVar(
            name="CONTEXT7_API_KEY",
            description="Context7 - Library docs (works without key, key removes rate limits)",
            requirement=Requirement.OPTIONAL,
            url="https://context7.com/",
            validator=validate_nonempty,
        ),
        ConfigVar(
            name="KAGI_API_KEY",
            description="Kagi Search - Premium search",
            requirement=Requirement.OPTIONAL,
            url="https://kagi.com/settings?p=api",
            validator=validate_nonempty,
        ),
        ConfigVar(
            name="SEARXNG_URL",
            description="SearXNG - Self-hosted meta search URL",
            requirement=Requirement.OPTIONAL,
            url="",
            validator=validate_url,
            secret=False,
        ),
        ConfigVar(
            name="NEO4J_URI",
            description="Neo4j URI for code graph queries",
            requirement=Requirement.OPTIONAL,
            default="bolt://localhost:7687",
            validator=validate_bolt_url,
            secret=False,
            group="neo4j",
        ),
        ConfigVar(
            name="NEO4J_USERNAME",
            description="Neo4j username",
            requirement=Requirement.OPTIONAL,
            default="neo4j",
            validator=validate_nonempty,
            secret=False,
            group="neo4j",
        ),
        ConfigVar(
            name="NEO4J_PASSWORD",
            description="Neo4j password",
            requirement=Requirement.OPTIONAL,
            validator=validate_nonempty,
            group="neo4j",
        ),
    ],
)

INLAND_EMPIRE_CONFIG = SkillConfig(
    name="inland-empire",
    display_name="Inland Empire",
    description="Memory substrate (works with local storage by default)",
    variables=[
        ConfigVar(
            name="LIBSQL_URL",
            description="LibSQL/Turso URL for fact memory",
            requirement=Requirement.OPTIONAL,
            url="https://turso.tech/",
            default="file:~/.inland-empire/construct.db",
            validator=validate_nonempty,
            secret=False,
        ),
        ConfigVar(
            name="LIBSQL_AUTH_TOKEN",
            description="LibSQL/Turso auth token (for remote)",
            requirement=Requirement.OPTIONAL,
            validator=validate_nonempty,
        ),
        ConfigVar(
            name="MEM0_API_KEY",
            description="Mem0 API for pattern memory",
            requirement=Requirement.OPTIONAL,
            url="https://app.mem0.ai/",
            validator=validate_nonempty,
        ),
        ConfigVar(
            name="POSTGRES_URL",
            description="PostgreSQL URL for self-hosted Mem0",
            requirement=Requirement.OPTIONAL,
            validator=validate_postgres_url,
        ),
        ConfigVar(
            name="INLAND_EMPIRE_STATE_DIR",
            description="State directory for memory files",
            requirement=Requirement.OPTIONAL,
            default="~/.inland-empire",
            validator=validate_nonempty,
            secret=False,
        ),
    ],
)

RHETORIC_CONFIG = SkillConfig(
    name="rhetoric",
    display_name="Rhetoric",
    description="Reasoning engine (needs 2+ LLM keys for deliberation)",
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
        ),
        ConfigVar(
            name="ANTHROPIC_API_KEY",
            description="Anthropic Claude API",
            requirement=Requirement.RECOMMENDED,
            url="https://console.anthropic.com/settings/keys",
            validator=validate_api_key_format("sk-ant-"),
            group="llm",
        ),
        ConfigVar(
            name="OPENROUTER_API_KEY",
            description="OpenRouter API (access to many models)",
            requirement=Requirement.RECOMMENDED,
            url="https://openrouter.ai/keys",
            validator=validate_api_key_format("sk-or-"),
            group="llm",
        ),
        ConfigVar(
            name="GOOGLE_CLOUD_API_KEY",
            description="Google Cloud / Gemini API",
            requirement=Requirement.OPTIONAL,
            url="https://console.cloud.google.com/apis/credentials",
            validator=validate_api_key_format("AIza"),
            group="llm",
        ),
        ConfigVar(
            name="OLLAMA_URL",
            description="Ollama local server URL",
            requirement=Requirement.OPTIONAL,
            url="https://ollama.ai",
            default="http://localhost:11434",
            validator=validate_url,
            secret=False,
            group="llm_local",
        ),
        ConfigVar(
            name="LMSTUDIO_URL",
            description="LM Studio local server URL",
            requirement=Requirement.OPTIONAL,
            url="https://lmstudio.ai",
            default="http://localhost:1234",
            validator=validate_url,
            secret=False,
            group="llm_local",
        ),
        ConfigVar(
            name="DEFAULT_LLM_PROVIDER",
            description="Default LLM provider for VibeCheck",
            requirement=Requirement.OPTIONAL,
            default="gemini",
            validator=validate_nonempty,
            secret=False,
        ),
    ],
)

VOLITION_CONFIG = SkillConfig(
    name="volition",
    display_name="Volition",
    description="Agency and execution (needs at least 1 LLM key)",
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
        ),
        ConfigVar(
            name="ANTHROPIC_API_KEY",
            description="Anthropic Claude API",
            requirement=Requirement.RECOMMENDED,
            url="https://console.anthropic.com/settings/keys",
            validator=validate_api_key_format("sk-ant-"),
            group="llm",
        ),
        ConfigVar(
            name="DEEPSEEK_API_KEY",
            description="DeepSeek API (code-specialized)",
            requirement=Requirement.OPTIONAL,
            url="https://platform.deepseek.com/",
            validator=validate_nonempty,
            group="llm",
        ),
        ConfigVar(
            name="SHODAN_API_KEY",
            description="Shodan API for security reconnaissance",
            requirement=Requirement.OPTIONAL,
            url="https://account.shodan.io/",
            validator=validate_nonempty,
        ),
    ],
)

ALL_CONFIGS = [ENCYCLOPEDIA_CONFIG, INLAND_EMPIRE_CONFIG, RHETORIC_CONFIG, VOLITION_CONFIG]


class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * len(text)}{Colors.RESET}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


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


def save_env_file(path: Path, env: dict[str, str], comments: dict[str, str] | None = None):
    """Save environment variables to a .env file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if comments:
        for key, comment in comments.items():
            if key in env:
                lines.append(f"# {comment}")
                lines.append(f"{key}={env[key]}")
                lines.append("")

    # Add any remaining vars not in comments
    for key, value in env.items():
        if comments and key in comments:
            continue
        lines.append(f"{key}={value}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def get_skill_env_path(skill_name: str) -> Path:
    """Get the .env.local path for a skill."""
    return CONSTRUCT_ROOT / skill_name / ".env.local"


def load_all_config() -> dict[str, dict[str, str]]:
    """Load configuration from all skill .env.local files."""
    config = {}
    for skill in SKILLS:
        env_path = get_skill_env_path(skill)
        config[skill] = load_env_file(env_path)
    return config


def get_current_value(var_name: str, all_config: dict[str, dict[str, str]]) -> str | None:
    """Get current value for a variable, checking env vars and .env.local files."""
    # First check environment
    if var_name in os.environ:
        return os.environ[var_name]

    # Then check all skill configs (many vars are shared)
    for skill_config in all_config.values():
        if var_name in skill_config:
            return skill_config[var_name]

    return None


def check_skill_status(skill_config: SkillConfig, all_config: dict[str, dict[str, str]]) -> tuple[bool, list[str], list[str]]:
    """
    Check if a skill is properly configured.
    Returns (is_ready, configured_vars, missing_required_vars)
    """
    configured = []
    missing = []
    group_coverage = {}

    for var in skill_config.variables:
        value = get_current_value(var.name, all_config)
        if value:
            configured.append(var.name)
            if var.group:
                group_coverage[var.group] = True
        elif var.requirement == Requirement.REQUIRED:
            missing.append(var.name)

    # Check required groups
    for group in skill_config.required_groups:
        if group not in group_coverage:
            # Find a var in this group to suggest
            for var in skill_config.variables:
                if var.group == group:
                    if var.name not in missing:
                        missing.append(f"(one of {group} group)")
                    break

    # Check minimum required
    is_ready = len(missing) == 0
    if skill_config.min_required > 0:
        llm_count = sum(1 for var in skill_config.variables
                       if var.group == "llm" and get_current_value(var.name, all_config))
        if llm_count < skill_config.min_required:
            is_ready = False
            if not any("llm" in m for m in missing):
                missing.append(f"(need {skill_config.min_required}+ LLM keys)")

    return is_ready, configured, missing


def show_status():
    """Display current configuration status for all skills."""
    print_header("Cognitive Construct Configuration Status")

    all_config = load_all_config()

    for skill_config in ALL_CONFIGS:
        is_ready, configured, missing = check_skill_status(skill_config, all_config)

        status_icon = f"{Colors.GREEN}✓{Colors.RESET}" if is_ready else f"{Colors.YELLOW}○{Colors.RESET}"
        print(f"\n{status_icon} {Colors.BOLD}{skill_config.display_name}{Colors.RESET} - {skill_config.description}")

        if configured:
            print(f"  {Colors.GREEN}Configured:{Colors.RESET} {', '.join(configured)}")
        if missing:
            print(f"  {Colors.YELLOW}Missing:{Colors.RESET} {', '.join(missing)}")
        if not configured and not missing:
            print(f"  {Colors.DIM}No configuration required (uses local defaults){Colors.RESET}")

    print()


def validate_config():
    """Validate all existing configuration."""
    print_header("Validating Configuration")

    all_config = load_all_config()
    errors = []
    warnings = []

    for skill_config in ALL_CONFIGS:
        print_subheader(skill_config.display_name)

        for var in skill_config.variables:
            value = get_current_value(var.name, all_config)
            if not value:
                if var.requirement == Requirement.REQUIRED:
                    print_error(f"{var.name}: Not set (required)")
                    errors.append(f"{skill_config.name}: {var.name} not set")
                continue

            if var.validator:
                is_valid, msg = var.validator(value)
                if is_valid:
                    print_success(f"{var.name}: {msg}")
                else:
                    print_error(f"{var.name}: {msg}")
                    errors.append(f"{skill_config.name}: {var.name} - {msg}")
            else:
                print_success(f"{var.name}: Set")

    print()
    if errors:
        print_error(f"Found {len(errors)} validation error(s)")
        return False
    else:
        print_success("All configured values are valid")
        return True


def export_shell():
    """Export configuration as shell commands."""
    print_header("Shell Export Commands")

    all_config = load_all_config()
    exported = set()

    print("# Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.)")
    print("# Or run: eval $(python scripts/setup.py --export)")
    print()

    for skill_config in ALL_CONFIGS:
        print(f"# {skill_config.display_name}")
        for var in skill_config.variables:
            if var.name in exported:
                continue
            value = get_current_value(var.name, all_config)
            if value:
                # Escape special characters for shell
                escaped = value.replace("'", "'\"'\"'")
                print(f"export {var.name}='{escaped}'")
                exported.add(var.name)
        print()


def prompt_value(var: ConfigVar, current: str | None) -> str | None:
    """Prompt user for a configuration value."""
    req_label = {
        Requirement.REQUIRED: f"{Colors.RED}required{Colors.RESET}",
        Requirement.RECOMMENDED: f"{Colors.YELLOW}recommended{Colors.RESET}",
        Requirement.OPTIONAL: f"{Colors.DIM}optional{Colors.RESET}",
    }[var.requirement]

    print(f"\n{Colors.BOLD}{var.name}{Colors.RESET} [{req_label}]")
    print(f"  {var.description}")
    if var.url:
        print(f"  {Colors.BLUE}Get key at: {var.url}{Colors.RESET}")

    if current:
        masked = current[:4] + "..." + current[-4:] if var.secret and len(current) > 12 else current
        print(f"  {Colors.GREEN}Current: {masked}{Colors.RESET}")

    default_prompt = f" [{var.default}]" if var.default else ""
    prompt_text = f"  Enter value{default_prompt} (or 'skip'): "

    try:
        if var.secret:
            import getpass
            value = getpass.getpass(prompt_text)
        else:
            value = input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    value = value.strip()

    if value.lower() == "skip" or value == "":
        if var.default and value == "":
            return var.default
        return None

    # Validate
    if var.validator:
        is_valid, msg = var.validator(value)
        if not is_valid:
            print_error(f"  {msg}")
            retry = input("  Try again? [Y/n]: ").strip().lower()
            if retry != "n":
                return prompt_value(var, current)
            return None
        print_success(f"  {msg}")

    return value


def configure_skill(skill_config: SkillConfig, all_config: dict[str, dict[str, str]]):
    """Interactively configure a single skill."""
    print_subheader(f"Configuring {skill_config.display_name}")
    print(f"{Colors.DIM}{skill_config.description}{Colors.RESET}")

    env_path = get_skill_env_path(skill_config.name)
    current_env = all_config.get(skill_config.name, {})
    new_env = dict(current_env)

    for var in skill_config.variables:
        current = get_current_value(var.name, all_config)
        value = prompt_value(var, current)

        if value is not None:
            new_env[var.name] = value
            # Also update all_config so subsequent prompts see it
            for skill in SKILLS:
                if skill not in all_config:
                    all_config[skill] = {}

    # Save
    if new_env != current_env:
        save_env_file(env_path, new_env)
        print_success(f"Saved to {env_path}")
    else:
        print_info("No changes made")


def interactive_setup():
    """Run interactive setup for all skills."""
    print_header("Cognitive Construct Setup")
    print("This wizard will help you configure all four skills.")
    print("Press Ctrl+C at any time to exit.\n")

    all_config = load_all_config()

    # Show current status
    print_subheader("Current Status")
    for skill_config in ALL_CONFIGS:
        is_ready, configured, missing = check_skill_status(skill_config, all_config)
        status = f"{Colors.GREEN}Ready{Colors.RESET}" if is_ready else f"{Colors.YELLOW}Needs setup{Colors.RESET}"
        print(f"  {skill_config.display_name}: {status}")

    print()

    # Menu
    while True:
        print_subheader("Select a skill to configure")
        print("  1. Encyclopedia (knowledge retrieval)")
        print("  2. Inland Empire (memory)")
        print("  3. Rhetoric (reasoning)")
        print("  4. Volition (agency)")
        print("  5. Configure all")
        print("  6. Show status")
        print("  7. Validate configuration")
        print("  8. Export shell commands")
        print("  9. Exit")

        try:
            choice = input(f"\n{Colors.CYAN}Enter choice [1-9]:{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if choice == "1":
            configure_skill(ENCYCLOPEDIA_CONFIG, all_config)
        elif choice == "2":
            configure_skill(INLAND_EMPIRE_CONFIG, all_config)
        elif choice == "3":
            configure_skill(RHETORIC_CONFIG, all_config)
        elif choice == "4":
            configure_skill(VOLITION_CONFIG, all_config)
        elif choice == "5":
            for config in ALL_CONFIGS:
                configure_skill(config, all_config)
        elif choice == "6":
            show_status()
        elif choice == "7":
            validate_config()
        elif choice == "8":
            export_shell()
        elif choice == "9":
            break
        else:
            print_warning("Invalid choice")


def test_api_connectivity():
    """Test API connectivity for configured services."""
    print_header("Testing API Connectivity")

    all_config = load_all_config()

    # Test OpenAI
    openai_key = get_current_value("OPENAI_API_KEY", all_config)
    if openai_key:
        print_info("Testing OpenAI API...")
        try:
            import httpx
            resp = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                print_success("OpenAI API: Connected")
            else:
                print_error(f"OpenAI API: Error {resp.status_code}")
        except Exception as e:
            print_error(f"OpenAI API: {e}")

    # Test Anthropic
    anthropic_key = get_current_value("ANTHROPIC_API_KEY", all_config)
    if anthropic_key:
        print_info("Testing Anthropic API...")
        try:
            import httpx
            resp = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                print_success("Anthropic API: Connected")
            else:
                print_error(f"Anthropic API: Error {resp.status_code}")
        except Exception as e:
            print_error(f"Anthropic API: {e}")

    # Test Exa
    exa_key = get_current_value("EXA_API_KEY", all_config)
    if exa_key:
        print_info("Testing Exa API...")
        try:
            import httpx
            resp = httpx.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": exa_key},
                json={"query": "test", "numResults": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                print_success("Exa API: Connected")
            else:
                print_error(f"Exa API: Error {resp.status_code}")
        except Exception as e:
            print_error(f"Exa API: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cognitive Construct Configuration Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup.py              # Interactive setup
  python scripts/setup.py --status     # Show configuration status
  python scripts/setup.py --validate   # Validate existing configuration
  python scripts/setup.py --export     # Export as shell commands
  python scripts/setup.py --test       # Test API connectivity
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show configuration status")
    parser.add_argument("--validate", action="store_true", help="Validate existing configuration")
    parser.add_argument("--export", action="store_true", help="Export as shell commands")
    parser.add_argument("--test", action="store_true", help="Test API connectivity")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.validate:
        sys.exit(0 if validate_config() else 1)
    elif args.export:
        export_shell()
    elif args.test:
        test_api_connectivity()
    else:
        interactive_setup()


if __name__ == "__main__":
    main()

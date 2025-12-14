"""OpenAI Web Search CLI - CLI-first tool for web search with reasoning models."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

import click

from .client import OpenAIWebSearchClient
from .config import (
    DEFAULT_MODEL_ENV_VAR,
    REASONING_MODELS,
    VALID_MODELS,
)
from .types import (
    ReasoningEffort,
    SearchContextSize,
    SearchType,
    UserLocation,
)


def web_search(
    query: str,
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    search_type: str = "web_search_preview",
    search_context_size: str = "medium",
    user_location: Optional[UserLocation] = None,
) -> str:
    """Execute web search using OpenAI's web search API.

    Args:
        query: The search query or question.
        model: AI model to use (default: OPENAI_DEFAULT_MODEL env var or gpt-5-mini).
        reasoning_effort: Effort level for reasoning models (low, medium, high, minimal).
        search_type: Web search API version.
        search_context_size: Context amount (low, medium, high).
        user_location: Optional location for localized results.

    Returns:
        Search result text from OpenAI.
    """
    if model is None:
        model = os.getenv(DEFAULT_MODEL_ENV_VAR, "gpt-5-mini")

    client = OpenAIWebSearchClient()

    async def _run_search():
        return await client.search(
            query=query,
            model=model,
            reasoning_effort=reasoning_effort,
            search_type=search_type,
            search_context_size=search_context_size,
            user_location=user_location,
        )

    result = asyncio.run(_run_search())

    if result.is_err():
        raise click.ClickException(f"Search failed: {result.error.message}")

    return result.value.content


@click.group()
@click.version_option(version="0.5.0", prog_name="openai-websearch")
def cli() -> None:
    """OpenAI Web Search CLI - Intelligent web search with reasoning models.

    For quick multi-round searches: Use 'gpt-5-mini' with --effort=low for fast iterations.

    For deep research: Use 'gpt-5' with --effort=medium or --effort=high.
    The result is already multi-round reasoned, so agents don't need continuous iterations.
    """


@cli.command()
@click.argument("query")
@click.option(
    "-m",
    "--model",
    type=click.Choice(sorted(list(VALID_MODELS))),
    default=None,
    help="AI model to use. Defaults to OPENAI_DEFAULT_MODEL env var or gpt-5-mini.",
)
@click.option(
    "-e",
    "--effort",
    "reasoning_effort",
    type=click.Choice([e.value for e in ReasoningEffort]),
    default=None,
    help="Reasoning effort level for supported models. Default: low for gpt-5-mini, medium for others.",
)
@click.option(
    "--type",
    "search_type",
    type=click.Choice([t.value for t in SearchType]),
    default=SearchType.PREVIEW.value,
    help="Web search API version.",
)
@click.option(
    "--context",
    "search_context_size",
    type=click.Choice([s.value for s in SearchContextSize]),
    default=SearchContextSize.MEDIUM.value,
    help="Amount of context to include in search results.",
)
@click.option(
    "--city",
    default=None,
    help="City for localized search results.",
)
@click.option(
    "--country",
    default=None,
    help="Country for localized search results.",
)
@click.option(
    "--region",
    default=None,
    help="Region for localized search results.",
)
@click.option(
    "--timezone",
    default=None,
    help="Timezone for localized search results (e.g., America/New_York).",
)
@click.option(
    "-o",
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
def search(
    query: str,
    model: Optional[str],
    reasoning_effort: Optional[str],
    search_type: str,
    search_context_size: str,
    city: Optional[str],
    country: Optional[str],
    region: Optional[str],
    timezone: Optional[str],
    output: str,
) -> None:
    """Search the web using OpenAI's reasoning models.

    QUERY is the search query or question to search for.

    Examples:

        openai-websearch search "latest AI news"

        openai-websearch search "quantum computing advances" -m gpt-5 -e high

        openai-websearch search "local events" --city "San Francisco" --country "US"
    """
    # Build user location if any location params provided
    user_location = None
    if city:
        if not timezone:
            raise click.ClickException("--timezone is required when --city is provided")
        user_location = UserLocation(
            city=city,
            country=country,
            region=region,
            timezone=timezone,
        )

    result_text = web_search(
        query,
        model=model,
        reasoning_effort=reasoning_effort,
        search_type=search_type,
        search_context_size=search_context_size,
        user_location=user_location,
    )

    if output == "json":
        click.echo(
            json.dumps(
                {
                    "query": query,
                    "model": model or os.getenv(DEFAULT_MODEL_ENV_VAR, "gpt-5-mini"),
                    "result": result_text,
                },
                indent=2,
            )
        )
    else:
        click.echo(result_text)


@cli.command()
@click.option(
    "--api-key",
    default=None,
    help="OpenAI API key. If not provided, prompts interactively.",
)
@click.option(
    "--default-model",
    default="gpt-5-mini",
    help="Default model to use.",
)
def install(api_key: Optional[str], default_model: str) -> None:
    """Install this tool in Claude Desktop configuration.

    Configures Claude Desktop to use this CLI as an MCP server.
    """
    import getpass
    import platform
    from pathlib import Path
    from shutil import which

    # Determine config path
    if sys.platform == "win32":
        config_dir = Path.home() / "AppData" / "Roaming" / "Claude"
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        raise click.ClickException(
            "Claude Desktop config not supported on this platform. "
            "Use manual configuration for Claude Code or Cursor."
        )

    if not config_dir.exists():
        raise click.ClickException(
            "Claude Desktop config directory not found. "
            "Please ensure Claude Desktop is installed and has been run at least once."
        )

    # Get API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    while not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        # We cannot easily validate the key without importing OpenAI client 
        # and making a call, but we removed the dependency.
        # We could use our new client to validate, but for now we just accept it.

    # Build environment
    local_bin = Path.home() / ".local" / "bin"
    pyenv_shims = Path.home() / ".pyenv" / "shims"
    python_version = platform.python_version()
    python_bin = Path.home() / "Library" / "Python" / python_version / "bin"
    path = os.environ["PATH"]

    sep = ";" if sys.platform == "win32" else ":"
    env_path = sep.join(str(p) for p in [local_bin, pyenv_shims, python_bin, path])

    env_dict = {
        "PATH": env_path,
        "OPENAI_API_KEY": api_key,
        "OPENAI_DEFAULT_MODEL": default_model,
    }

    # Find uvx
    uv = which("uvx", path=env_path)
    command = uv if uv else "uvx"

    # Update config
    config_file = config_dir / "claude_desktop_config.json"
    if not config_file.exists():
        config_file.write_text("{}")

    config = json.loads(config_file.read_text())
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Preserve existing env vars
    server_name = "openai-websearch-mcp"
    if server_name in config["mcpServers"] and "env" in config["mcpServers"][server_name]:
        existing_env = config["mcpServers"][server_name]["env"]
        env_dict = {**existing_env, **env_dict}

    config["mcpServers"][server_name] = {
        "command": command,
        "args": [server_name],
        "env": env_dict,
    }

    config_file.write_text(json.dumps(config, indent=2))
    click.echo(f"Successfully installed {server_name} in Claude Desktop")
    click.echo(f"Config: {config_file}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
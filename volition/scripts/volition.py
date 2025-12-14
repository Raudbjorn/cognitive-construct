#!/usr/bin/env python3
"""
Volition: The Will to Act

Executive skill entrypoint for semantic code editing, LLM consultation,
web search, and security reconnaissance.

Usage:
    python3 volition.py act "refactor authentication for security"
    python3 volition.py edit "UserAuth.validate" "add rate limiting"
    python3 volition.py query web "Python 3.13 features"
    python3 volition.py query llm "JWT rotation best practices" --tag coding
    python3 volition.py query security "exposed MongoDB" --confirm
    python3 volition.py capabilities
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOLITION_DIR = Path.home() / ".volition"
AUDIT_LOG = VOLITION_DIR / "audit.log"
SHODAN_AUDIT_LOG = VOLITION_DIR / "shodan_audit.jsonl"
RATE_LIMITS_FILE = VOLITION_DIR / "rate_limits.json"

SHODAN_RATE_LIMIT_PER_HOUR = 10
IP_REDACTION_PATTERN = re.compile(r"(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}")

# Intent classification keywords
INTENT_KEYWORDS: dict[str, list[str]] = {
    "code_edit": ["refactor", "edit", "modify", "add", "remove", "fix", "update", "change"],
    "llm_call": ["explain", "analyze", "review", "suggest", "consult", "help", "advise"],
    "web_search": ["search", "find", "lookup", "what is", "latest", "current", "news"],
    "security": ["scan", "expose", "vulnerability", "shodan", "security", "recon"],
}


class ExitCode:
    """Exit codes for CLI commands."""

    SUCCESS = 0
    INVALID_INPUT = 1
    RESOURCE_NOT_FOUND = 2
    BACKEND_UNAVAILABLE = 3
    PERMISSION_DENIED = 4


# ---------------------------------------------------------------------------
# Result Type (inline for standalone operation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Success case."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Err[E]:
    """Error case."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


type Result[T, E] = Ok[T] | Err[E]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VolitionError:
    """Error from Volition operations."""

    message: str
    code: int = ExitCode.BACKEND_UNAVAILABLE
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result from an action."""

    status: Literal["success", "error", "confirmation_required"]
    handler: str | None = None
    summary: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """Status of a capability."""

    status: Literal["available", "unavailable", "restricted"]
    backend: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------


def _ensure_volition_dir() -> None:
    """Ensure ~/.volition directory exists."""
    VOLITION_DIR.mkdir(parents=True, exist_ok=True)


def _load_env_local() -> None:
    """Load environment variables from .env.local if present."""
    env_file = Path(__file__).parent / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key.strip(), value)


def _is_shodan_disabled() -> bool:
    """Check if Shodan is disabled via environment."""
    return (
        os.environ.get("VOLITION_DISABLE_SHODAN", "").lower() == "true"
        or os.environ.get("NOCP_FLAG_SHODAN_ENABLED", "").lower() == "false"
    )


def _has_api_key(env_var: str) -> bool:
    """Check if an API key is set."""
    return bool(os.environ.get(env_var))


# ---------------------------------------------------------------------------
# Rate Limiting (R.23.1)
# ---------------------------------------------------------------------------


def _load_rate_limits() -> dict[str, Any]:
    """Load rate limit state."""
    if RATE_LIMITS_FILE.exists():
        try:
            with open(RATE_LIMITS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"shodan": {"count": 0, "window_start": 0}}


def _save_rate_limits(state: dict[str, Any]) -> None:
    """Save rate limit state."""
    _ensure_volition_dir()
    with open(RATE_LIMITS_FILE, "w") as f:
        json.dump(state, f)


def _check_shodan_rate_limit() -> Result[None, VolitionError]:
    """Check and update Shodan rate limit."""
    state = _load_rate_limits()
    shodan = state.get("shodan", {"count": 0, "window_start": 0})

    now = time.time()
    window_start = shodan.get("window_start", 0)
    count = shodan.get("count", 0)

    # Reset window if hour has passed
    if now - window_start > 3600:
        shodan = {"count": 0, "window_start": now}
        state["shodan"] = shodan
        _save_rate_limits(state)

    if shodan["count"] >= SHODAN_RATE_LIMIT_PER_HOUR:
        remaining = int(3600 - (now - window_start))
        return Err(
            VolitionError(
                message=f"Shodan rate limit exceeded ({SHODAN_RATE_LIMIT_PER_HOUR}/hour). "
                f"Try again in {remaining}s.",
                code=ExitCode.PERMISSION_DENIED,
            )
        )

    # Increment counter
    shodan["count"] = count + 1
    state["shodan"] = shodan
    _save_rate_limits(state)
    return Ok(None)


# ---------------------------------------------------------------------------
# Audit Logging (R.22.4)
# ---------------------------------------------------------------------------


def _audit_log(entry: dict[str, Any]) -> None:
    """Append to general audit log."""
    _ensure_volition_dir()
    entry["timestamp"] = datetime.now().isoformat()
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _shodan_audit_log(
    query: str,
    target: str | None,
    result_count: int,
    severity: str = "INFO",
) -> None:
    """Append to Shodan-specific audit log with severity (R.22.4)."""
    _ensure_volition_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "requester": os.environ.get("USER", "unknown"),
        "query": query,
        "target": target,
        "result_count": result_count,
        "severity": severity,
    }
    with open(SHODAN_AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data Redaction (R.22.3)
# ---------------------------------------------------------------------------


def _redact_ip(ip: str) -> str:
    """Partially redact IP address."""
    match = IP_REDACTION_PATTERN.match(ip)
    if match:
        return f"{match.group(1)}.xxx.xxx"
    return ip


def _redact_shodan_results(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive data from Shodan results."""
    if "matches" in data:
        for match in data["matches"]:
            if "ip_str" in match:
                match["ip"] = _redact_ip(match.pop("ip_str"))
    if "ip_str" in data:
        data["ip"] = _redact_ip(data.pop("ip_str"))
    return data


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------


def _classify_intent(action: str) -> str:
    """Classify action intent based on keywords."""
    action_lower = action.lower()
    scores = {category: 0 for category in INTENT_KEYWORDS}

    for category, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in action_lower:
                scores[category] += 1

    # Return category with highest score, default to llm_call
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "llm_call"


# ---------------------------------------------------------------------------
# Backend Handlers
# ---------------------------------------------------------------------------


async def _handle_code_edit(
    symbol: str,
    change: str,
    project: str = ".",
    fallback: bool = False,
) -> Result[ActionResult, VolitionError]:
    """Handle semantic code editing via Serena."""
    # Check if serena is available
    serena_path = Path(__file__).parent / "serena"
    if not serena_path.exists():
        return Err(
            VolitionError(
                message="Serena backend not available",
                code=ExitCode.BACKEND_UNAVAILABLE,
            )
        )

    # TODO: Integrate with Serena's symbol editing tools
    # For now, return a placeholder indicating the capability exists
    return Ok(
        ActionResult(
            status="success",
            handler="code_edit",
            summary=f"Would edit symbol '{symbol}' with change: {change}",
            data={"symbol": symbol, "project": project, "fallback": fallback},
        )
    )


async def _handle_llm_call(
    prompt: str,
    provider: str | None = None,
    tag: str = "general",
) -> Result[ActionResult, VolitionError]:
    """Handle LLM consultation via cross-llm-mcp."""
    try:
        # Attempt to import the client
        sys.path.insert(0, str(Path(__file__).parent / "cross-llm-mcp" / "src"))
        from cross_llm_mcp import CrossLLMClient

        client = CrossLLMClient()
        available = client.get_available_providers()

        if not available:
            return Err(
                VolitionError(
                    message="No LLM providers configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY.",
                    code=ExitCode.BACKEND_UNAVAILABLE,
                )
            )

        result = await client.call(prompt, provider=provider, tag=tag)  # type: ignore

        if result.is_err():
            return Err(
                VolitionError(
                    message=result.error.message,
                    code=ExitCode.BACKEND_UNAVAILABLE,
                )
            )

        _audit_log({"action": "llm_call", "tag": tag, "provider": result.value.provider})

        return Ok(
            ActionResult(
                status="success",
                handler="llm_call",
                summary=result.value.response[:200] + "..." if len(result.value.response) > 200 else result.value.response,
                data={
                    "provider": result.value.provider,
                    "model": result.value.model,
                    "response": result.value.response,
                },
            )
        )

    except ImportError:
        return Err(
            VolitionError(
                message="cross-llm-mcp backend not available",
                code=ExitCode.BACKEND_UNAVAILABLE,
            )
        )


async def _handle_web_search(query: str) -> Result[ActionResult, VolitionError]:
    """Handle web search via openai-websearch-mcp."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "openai-websearch-mcp" / "src"))
        from openai_websearch_mcp import OpenAIWebSearchClient

        client = OpenAIWebSearchClient()
        result = await client.search(query)

        if result.is_err():
            return Err(
                VolitionError(
                    message=result.error.message,
                    code=ExitCode.BACKEND_UNAVAILABLE,
                )
            )

        _audit_log({"action": "web_search", "query": query})

        return Ok(
            ActionResult(
                status="success",
                handler="web_search",
                summary=result.value.content[:200] + "..." if len(result.value.content) > 200 else result.value.content,
                data={"content": result.value.content},
            )
        )

    except ImportError:
        return Err(
            VolitionError(
                message="openai-websearch-mcp backend not available",
                code=ExitCode.BACKEND_UNAVAILABLE,
            )
        )


async def _handle_security_query(
    query: str,
    confirm: bool = False,
) -> Result[ActionResult, VolitionError]:
    """Handle security query via mcp-shodan with safety constraints (R.22-R.23)."""
    # R.22.2: Check if disabled
    if _is_shodan_disabled():
        return Err(
            VolitionError(
                message="Shodan queries disabled via VOLITION_DISABLE_SHODAN or NOCP_FLAG_SHODAN_ENABLED=false",
                code=ExitCode.PERMISSION_DENIED,
            )
        )

    # R.22.1: Require explicit confirmation
    if not confirm:
        return Ok(
            ActionResult(
                status="confirmation_required",
                handler="security",
                summary="Security queries require explicit confirmation. Use --confirm flag.",
                data={"query": query, "action_required": "--confirm"},
            )
        )

    # R.23.1: Rate limiting
    rate_check = _check_shodan_rate_limit()
    if rate_check.is_err():
        return Err(rate_check.error)

    try:
        sys.path.insert(0, str(Path(__file__).parent / "mcp-shodan" / "src"))
        from mcp_shodan import ShodanClient

        client = ShodanClient()
        result = await client.search(query, limit=10)

        if result.is_err():
            _shodan_audit_log(query, None, 0, severity="ERROR")
            return Err(
                VolitionError(
                    message=result.error.message,
                    code=ExitCode.BACKEND_UNAVAILABLE,
                )
            )

        # R.22.3: Redact results
        raw_data = asdict(result.value)
        redacted = _redact_shodan_results(raw_data)

        # R.22.4: Audit log
        _shodan_audit_log(query, None, result.value.total, severity="INFO")

        return Ok(
            ActionResult(
                status="success",
                handler="security",
                summary=f"Found {result.value.total} results",
                data=redacted,
            )
        )

    except ImportError:
        return Err(
            VolitionError(
                message="mcp-shodan backend not available",
                code=ExitCode.BACKEND_UNAVAILABLE,
            )
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_act(action: str, handler: str | None = None) -> dict[str, Any]:
    """Execute a general action with automatic routing."""
    selected_handler = handler or _classify_intent(action)

    if selected_handler == "code_edit":
        # For act command, we need to extract symbol/change from natural language
        # This is a simplified implementation
        result = await _handle_llm_call(
            f"Analyze this request and suggest specific code changes: {action}",
            tag="coding",
        )
    elif selected_handler == "llm_call":
        result = await _handle_llm_call(action)
    elif selected_handler == "web_search":
        result = await _handle_web_search(action)
    elif selected_handler == "security":
        # Security requires explicit confirmation via query command
        result = Ok(
            ActionResult(
                status="error",
                handler="security",
                summary="Security queries require explicit confirmation. Use: query security <query> --confirm",
            )
        )
    else:
        result = await _handle_llm_call(action)

    if result.is_err():
        return {"status": "error", "code": result.error.code, "message": result.error.message}

    return asdict(result.value)


async def cmd_edit(symbol: str, change: str, project: str = ".", fallback: bool = False) -> dict[str, Any]:
    """Execute semantic code edit."""
    result = await _handle_code_edit(symbol, change, project, fallback)

    if result.is_err():
        return {"status": "error", "code": result.error.code, "message": result.error.message}

    return asdict(result.value)


async def cmd_query(service: str, query: str, **kwargs: Any) -> dict[str, Any]:
    """Query external services."""
    if service == "web":
        result = await _handle_web_search(query)
    elif service == "llm":
        result = await _handle_llm_call(
            query,
            provider=kwargs.get("provider"),
            tag=kwargs.get("tag", "general"),
        )
    elif service == "security":
        result = await _handle_security_query(query, confirm=kwargs.get("confirm", False))
    else:
        return {
            "status": "error",
            "code": ExitCode.INVALID_INPUT,
            "message": f"Unknown service: {service}. Valid: web, llm, security",
        }

    if result.is_err():
        return {"status": "error", "code": result.error.code, "message": result.error.message}

    return asdict(result.value)


def cmd_capabilities() -> dict[str, Any]:
    """List available capabilities and their status."""
    _load_env_local()

    capabilities: dict[str, CapabilityStatus] = {}

    # Code editing (Serena)
    serena_path = Path(__file__).parent / "serena"
    capabilities["code_editing"] = CapabilityStatus(
        status="available" if serena_path.exists() else "unavailable",
        backend="serena",
        details={"languages": ["python", "typescript", "rust", "go"]} if serena_path.exists() else {},
    )

    # LLM consultation
    llm_providers = []
    if _has_api_key("OPENAI_API_KEY"):
        llm_providers.append("openai")
    if _has_api_key("ANTHROPIC_API_KEY"):
        llm_providers.append("anthropic")
    if _has_api_key("DEEPSEEK_API_KEY"):
        llm_providers.append("deepseek")
    if _has_api_key("GEMINI_API_KEY"):
        llm_providers.append("gemini")
    if _has_api_key("XAI_API_KEY"):
        llm_providers.append("grok")
    if _has_api_key("MISTRAL_API_KEY"):
        llm_providers.append("mistral")

    capabilities["llm_consultation"] = CapabilityStatus(
        status="available" if llm_providers else "unavailable",
        backend="cross-llm-mcp",
        details={"providers": llm_providers},
    )

    # Web search
    capabilities["web_search"] = CapabilityStatus(
        status="available" if _has_api_key("OPENAI_API_KEY") else "unavailable",
        backend="openai-websearch",
        details={},
    )

    # Security queries
    if _is_shodan_disabled():
        sec_status: Literal["available", "unavailable", "restricted"] = "unavailable"
        sec_details: dict[str, Any] = {"reason": "disabled via environment"}
    elif _has_api_key("SHODAN_API_KEY"):
        sec_status = "restricted"
        sec_details = {"rate_limit": f"{SHODAN_RATE_LIMIT_PER_HOUR}/hour", "requires_confirmation": True}
    else:
        sec_status = "unavailable"
        sec_details = {"reason": "SHODAN_API_KEY not set"}

    capabilities["security_queries"] = CapabilityStatus(
        status=sec_status,
        backend="shodan",
        details=sec_details,
    )

    return {cap: asdict(status) for cap, status in capabilities.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Volition: The Will to Act - Executive skill for code editing, LLM consultation, and more.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # act command
    act_parser = subparsers.add_parser("act", help="Execute a general action")
    act_parser.add_argument("action", help="Action to perform (natural language)")
    act_parser.add_argument("--handler", choices=["code_edit", "llm_call", "web_search"], help="Override automatic routing")

    # edit command
    edit_parser = subparsers.add_parser("edit", help="Perform semantic code edit")
    edit_parser.add_argument("symbol", help="Symbol to edit (e.g., 'UserAuth.validate')")
    edit_parser.add_argument("change", help="Change to apply (natural language)")
    edit_parser.add_argument("--project", default=".", help="Project root directory")
    edit_parser.add_argument("--fallback", action="store_true", help="Allow text-based editing if LSP unavailable")

    # query command
    query_parser = subparsers.add_parser("query", help="Query external services")
    query_parser.add_argument("service", choices=["web", "llm", "security"], help="Service to query")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--provider", help="Override LLM provider (for llm service)")
    query_parser.add_argument("--tag", default="general", help="Task tag for LLM routing")
    query_parser.add_argument("--confirm", action="store_true", help="Confirm security queries (use with caution)")

    # capabilities command
    subparsers.add_parser("capabilities", help="List available capabilities")

    return parser


def _output_json(data: dict[str, Any], exit_code: int = 0) -> None:
    """Output JSON and exit."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(exit_code)


def main() -> None:
    """Main entry point."""
    _load_env_local()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "act":
            result = asyncio.run(cmd_act(args.action, handler=args.handler))
            code = result.get("code", ExitCode.SUCCESS if result.get("status") == "success" else 1)
            _output_json(result, code)

        elif args.command == "edit":
            result = asyncio.run(cmd_edit(args.symbol, args.change, args.project, args.fallback))
            code = result.get("code", ExitCode.SUCCESS if result.get("status") == "success" else 1)
            _output_json(result, code)

        elif args.command == "query":
            result = asyncio.run(
                cmd_query(
                    args.service,
                    args.query,
                    provider=args.provider,
                    tag=args.tag,
                    confirm=args.confirm,
                )
            )
            code = result.get("code", ExitCode.SUCCESS if result.get("status") == "success" else 1)
            _output_json(result, code)

        elif args.command == "capabilities":
            result = cmd_capabilities()
            _output_json(result, ExitCode.SUCCESS)

    except KeyboardInterrupt:
        _output_json({"status": "error", "code": ExitCode.INVALID_INPUT, "message": "Interrupted"}, ExitCode.INVALID_INPUT)
    except Exception as e:
        _output_json({"status": "error", "code": ExitCode.BACKEND_UNAVAILABLE, "message": str(e)}, ExitCode.BACKEND_UNAVAILABLE)


if __name__ == "__main__":
    main()

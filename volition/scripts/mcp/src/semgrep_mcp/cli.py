#!/usr/bin/env python3
"""
Semgrep CLI - Direct CLI wrapper for Semgrep security scanning.

Replaces SSE/HTTP MCP transport with direct semgrep invocation, exposing
functionality as simple CLI commands.

Commands:
- scan: Scan code files with Semgrep
- scan-remote: Scan code content provided directly
- languages: List supported languages
- status: Check Semgrep installation status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from semgrep_mcp.client import SemgrepClient
from semgrep_mcp.types import CodeFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env.local"


class ExitCode:
    """Exit codes for CLI commands."""

    SUCCESS = 0
    USER_ERROR = 1
    CONFIG_ERROR = 2
    BACKEND_ERROR = 3
    INTERNAL_ERROR = 4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_env_local() -> dict[str, str]:
    """Load environment variables from .env.local file."""
    env_vars: dict[str, str] = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip().strip("\"'")
                    env_vars[key.strip()] = value
                    os.environ[key.strip()] = value
    return env_vars


def _output_json(data: dict[str, Any], exit_code: int = 0) -> None:
    """Output JSON response and exit with code."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(exit_code)


def _output_error(code: int, message: str) -> None:
    """Output error response and exit."""
    _output_json({"status": "error", "code": code, "message": message}, code)


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a result to a JSON-serializable dictionary."""
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return dict(result) if hasattr(result, "items") else {"value": result}


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------


async def _handle_scan(args: argparse.Namespace) -> dict[str, Any]:
    """Handle scan command."""
    _load_env_local()

    client = SemgrepClient()
    result = await client.scan_files(paths=args.paths, config=args.config)

    if result.is_err():
        return {
            "status": "error",
            "code": ExitCode.BACKEND_ERROR,
            "message": result.error.message,
            "error_code": result.error.code,
        }

    scan_result = result.value
    findings = scan_result.results

    return {
        "status": "success",
        "version": scan_result.version,
        "findings_count": len(findings),
        "findings": [
            {
                "rule_id": f.get("check_id"),
                "path": f.get("path"),
                "line": f.get("start", {}).get("line"),
                "message": f.get("extra", {}).get("message"),
                "severity": f.get("extra", {}).get("severity"),
            }
            for f in findings
        ],
        "errors_count": len(scan_result.errors),
        "scanned_paths": scan_result.paths.get("scanned", []),
    }


async def _handle_scan_remote(args: argparse.Namespace) -> dict[str, Any]:
    """Handle scan-remote command (scan inline code)."""
    _load_env_local()

    # Parse files from JSON
    try:
        files_data = json.loads(args.files)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "code": ExitCode.USER_ERROR,
            "message": 'Invalid JSON format. Expected: [{"path": "file.py", "content": "..."}]',
        }

    # Convert to CodeFile objects
    files = [CodeFile.from_dict(f) for f in files_data]

    client = SemgrepClient()
    result = await client.scan_content(files=files, config=args.config)

    if result.is_err():
        return {
            "status": "error",
            "code": ExitCode.BACKEND_ERROR,
            "message": result.error.message,
            "error_code": result.error.code,
        }

    scan_result = result.value
    findings = scan_result.results

    return {
        "status": "success",
        "version": scan_result.version,
        "findings_count": len(findings),
        "findings": [
            {
                "rule_id": f.get("check_id"),
                "path": f.get("path"),
                "line": f.get("start", {}).get("line"),
                "message": f.get("extra", {}).get("message"),
                "severity": f.get("extra", {}).get("severity"),
            }
            for f in findings
        ],
        "errors_count": len(scan_result.errors),
        "note": "Paths shown are relative to temporary scan directory",
    }


async def _handle_languages(_args: argparse.Namespace) -> dict[str, Any]:
    """Handle languages command."""
    client = SemgrepClient()
    result = await client.get_supported_languages()

    if result.is_err():
        return {
            "status": "error",
            "code": ExitCode.BACKEND_ERROR,
            "message": result.error.message,
        }

    languages = result.value
    return {
        "status": "success",
        "languages": languages,
        "count": len(languages),
    }


async def _handle_status(_args: argparse.Namespace) -> dict[str, Any]:
    """Handle status command."""
    _load_env_local()

    client = SemgrepClient()
    version_result = await client.get_version()

    if version_result.is_err():
        return {
            "status": "success",
            "semgrep_installed": False,
            "semgrep_path": None,
            "version": None,
            "error": version_result.error.message,
            "available_commands": ["scan", "scan-remote", "languages"],
        }

    return {
        "status": "success",
        "semgrep_installed": True,
        "semgrep_path": client._resolved_path,
        "version": version_result.value,
        "available_commands": ["scan", "scan-remote", "languages"],
    }


# ---------------------------------------------------------------------------
# CLI Setup
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Semgrep CLI - Security scanning tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan local files with Semgrep",
        description="Scan files or directories for security issues using Semgrep.",
    )
    scan_parser.add_argument("paths", nargs="+", help="Paths to scan (files or directories)")
    scan_parser.add_argument(
        "--config",
        help="Semgrep config (e.g., 'auto', 'p/security', path to rules file)",
    )

    # Scan-remote command
    remote_parser = subparsers.add_parser(
        "scan-remote",
        help="Scan code content (not files)",
        description="Scan code content provided as JSON.",
    )
    remote_parser.add_argument(
        "files",
        help='JSON array of files: [{"path": "file.py", "content": "..."}]',
    )
    remote_parser.add_argument("--config", help="Semgrep config")

    # Languages command
    subparsers.add_parser(
        "languages",
        help="List supported languages",
        description="Show all programming languages supported by Semgrep.",
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Check Semgrep installation status",
        description="Check if Semgrep is installed and display version information.",
    )

    return parser


async def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch command to handler."""
    handlers = {
        "scan": _handle_scan,
        "scan-remote": _handle_scan_remote,
        "languages": _handle_languages,
        "status": _handle_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        return {
            "status": "error",
            "code": ExitCode.USER_ERROR,
            "message": f"Unknown command: {args.command}",
        }

    return await handler(args)


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = asyncio.run(_dispatch(args))
        exit_code = ExitCode.SUCCESS if result.get("status") == "success" else result.get("code", 1)
        _output_json(result, exit_code)
    except KeyboardInterrupt:
        _output_error(ExitCode.USER_ERROR, "Interrupted by user")
    except Exception as exc:
        _output_error(ExitCode.INTERNAL_ERROR, f"Internal error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

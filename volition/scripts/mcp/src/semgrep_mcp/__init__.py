"""
semgrep-mcp - Python client library for Semgrep security scanning.

Usage:
    from semgrep_mcp import SemgrepClient, scan_files, scan_content

    # Client-based usage (recommended for multiple calls)
    client = SemgrepClient()

    result = await client.scan_files(["/path/to/code"])
    if result.is_ok():
        for finding in result.value.results:
            print(f"{finding['path']}: {finding['check_id']}")
    else:
        print(f"Error: {result.error.message}")

    # Convenience function (for one-off calls)
    result = await scan_files(["/path/to/code"], config="p/security")

CLI:
    Run the CLI with: semgrep-cli scan /path/to/code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Core types - always available
from semgrep_mcp.result import Err, Ok, Result
from semgrep_mcp.types import (
    Assistant,
    Autofix,
    Autotriage,
    CodeFile,
    CodeWithLanguage,
    Component,
    Confidence,
    ExternalTicket,
    Finding,
    Guidance,
    Location,
    Repository,
    ReviewComment,
    Rule,
    SemgrepError,
    SemgrepScanResult,
    Severity,
    SourcingPolicy,
    TriageState,
)
from semgrep_mcp.version import __version__

# Lazy imports for client module
if TYPE_CHECKING:
    from semgrep_mcp.client import (
        SemgrepClient as SemgrepClient,
        get_version as get_version,
        scan_content as scan_content,
        scan_files as scan_files,
    )


def __getattr__(name: str):
    """Lazy import for client module."""
    if name == "SemgrepClient":
        from semgrep_mcp.client import SemgrepClient

        return SemgrepClient
    if name == "scan_files":
        from semgrep_mcp.client import scan_files

        return scan_files
    if name == "scan_content":
        from semgrep_mcp.client import scan_content

        return scan_content
    if name == "get_version":
        from semgrep_mcp.client import get_version

        return get_version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # noqa: RUF022 - intentional grouping per architecture guide
    # Version
    "__version__",
    # Client
    "SemgrepClient",
    # Convenience functions
    "scan_files",
    "scan_content",
    "get_version",
    # Request types
    "CodeFile",
    "CodeWithLanguage",
    # Response types
    "SemgrepScanResult",
    "Finding",
    "Location",
    "Repository",
    "Rule",
    "Assistant",
    "Autofix",
    "Autotriage",
    "Guidance",
    "Component",
    "ExternalTicket",
    "ReviewComment",
    "SourcingPolicy",
    # Enums
    "TriageState",
    "Severity",
    "Confidence",
    # Error types
    "SemgrepError",
    # Result types
    "Result",
    "Ok",
    "Err",
]

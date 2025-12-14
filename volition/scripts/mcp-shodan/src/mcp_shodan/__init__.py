"""
mcp-shodan - Shodan Security API Wrapper

Usage:
    from mcp_shodan import ShodanClient

    client = ShodanClient(api_key="...")  # or set SHODAN_API_KEY env var

    # Search Shodan
    result = await client.search("apache port:443")
    if result.is_ok():
        for match in result.value.matches:
            print(f"{match.ip_str}:{match.port} - {match.org}")
    else:
        print(f"Error: {result.error.message}")

    # Get host info
    result = await client.host("8.8.8.8")

    # CVE lookup
    result = await client.cve("CVE-2021-44228")
"""

from .client import ShodanClient
from .types import (
    ApiError,
    CveInfo,
    Location,
    ShodanHostResult,
    ShodanSearchMatch,
    ShodanSearchResult,
)
from .result import Result, Ok, Err
from .cli import main

__all__ = [
    # Client
    "ShodanClient",
    "main",
    # Types
    "ShodanSearchResult",
    "ShodanSearchMatch",
    "ShodanHostResult",
    "CveInfo",
    "Location",
    "ApiError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "0.1.0"

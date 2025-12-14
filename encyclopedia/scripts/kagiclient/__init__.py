"""
kagiclient - Python client library for Kagi Search and Summarizer API

Usage:
    from kagiclient import KagiClient

    client = KagiClient(api_key="your-key")  # or set KAGI_API_KEY env var

    # Search
    result = await client.search("latest python frameworks 2024")
    if result.is_ok():
        for r in result.value.results:
            print(f"{r.title}: {r.url}")

    # Summarize
    result = await client.summarize("https://example.com/article")
    if result.is_ok():
        print(result.value.summary)
"""

from .client import KagiClient, search, summarize
from .types import (
    ErrorCode,
    KagiError,
    SearchResponse,
    SearchResult,
    SummarizerEngine,
    SummaryResponse,
    SummaryType,
)
from .result import Result, Ok, Err

__all__ = [
    "KagiClient",
    "search",
    "summarize",
    "SearchResponse",
    "SearchResult",
    "SummaryResponse",
    "SummaryType",
    "SummarizerEngine",
    "KagiError",
    "ErrorCode",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

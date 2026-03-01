"""
kagiclient - Python client library for Kagi Search, FastGPT, Summarizer, and Enrich APIs.

Usage:
    from kagiclient import KagiClient

    client = KagiClient(api_key="your-key")  # or set KAGI_API_KEY env var

    # FastGPT (AI answer — no Search API beta needed)
    result = await client.fastgpt("explain Rust ownership")
    if result.is_ok():
        print(result.value.output)

    # Summarize
    result = await client.summarize("https://example.com/article")
    if result.is_ok():
        print(result.value.summary)
"""

from .client import KagiClient, search, fastgpt, summarize, enrich
from .types import (
    EnrichResponse,
    EnrichResult,
    ErrorCode,
    FastGPTReference,
    FastGPTResponse,
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
    "fastgpt",
    "summarize",
    "enrich",
    "EnrichResponse",
    "EnrichResult",
    "FastGPTReference",
    "FastGPTResponse",
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

__version__ = "2.0.0"

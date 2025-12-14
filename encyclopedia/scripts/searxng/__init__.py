"""
searxng - Python client library for SearXNG metasearch engine

Usage:
    from searxng import SearxngClient, search

    # Quick search (sync)
    result = search("python async frameworks")
    if result.is_ok():
        for r in result.value.results:
            print(f"{r.title}: {r.url}")

    # Client-based usage
    client = SearxngClient(base_url="http://localhost:8080")

    # Sync search
    result = client.search("python async frameworks", limit=5)

    # Async search
    result = await client.search_async("python async frameworks")

    # Format as text
    if result.is_ok():
        print(client.format_results(result.value))
"""

from .client import SearxngClient, search, search_async
from .types import Infobox, InfoboxUrl, SearchResponse, SearchResult, SearxngError
from .result import Result, Ok, Err

__all__ = [
    "SearxngClient",
    "search",
    "search_async",
    "SearchResponse",
    "SearchResult",
    "Infobox",
    "InfoboxUrl",
    "SearxngError",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

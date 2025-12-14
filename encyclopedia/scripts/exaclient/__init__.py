"""
exaclient - Python client library for Exa AI search API

Usage:
    from exaclient import ExaClient

    client = ExaClient(api_key="your-key")  # or set EXA_API_KEY env var

    # Web search
    result = client.web_search("latest python frameworks 2024")
    if result.is_ok():
        print(result.value.context)

    # Code search
    result = client.code_search("python async http client examples")
    if result.is_ok():
        print(result.value.content)
"""

from .client import ExaClient, web_search, code_search
from .types import (
    SearchType,
    LivecrawlMode,
    WebSearchOptions,
    CodeSearchOptions,
    WebSearchResult,
    CodeSearchResult,
    ExaSearchResponse,
    ExaCodeResponse,
    DeepResearchRequest,
    DeepResearchResponse,
)
from .result import Result, Ok, Err

__all__ = [
    "ExaClient",
    "web_search",
    "code_search",
    "SearchType",
    "LivecrawlMode",
    "WebSearchOptions",
    "CodeSearchOptions",
    "WebSearchResult",
    "CodeSearchResult",
    "ExaSearchResponse",
    "ExaCodeResponse",
    "DeepResearchRequest",
    "DeepResearchResponse",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

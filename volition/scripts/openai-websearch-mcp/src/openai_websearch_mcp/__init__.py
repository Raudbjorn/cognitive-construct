"""
openai-websearch-mcp - OpenAI Web Search client and CLI

Usage:
    from openai_websearch_mcp import OpenAIWebSearchClient

    client = OpenAIWebSearchClient(api_key="...")  # or set OPENAI_API_KEY env var

    result = await client.search("latest AI news")
    if result.is_ok():
        print(result.value.content)
    else:
        print(f"Error: {result.error.message}")

    # With reasoning model and effort level
    result = await client.search(
        "quantum computing advances",
        model="gpt-5",
        reasoning_effort="high"
    )

CLI:
    openai-websearch search "latest AI news" -m gpt-5-mini
"""

from .server import cli, main, web_search
from .client import OpenAIWebSearchClient
from .types import (
    ApiError,
    ReasoningEffort,
    SearchContextSize,
    SearchResponse,
    SearchType,
    UserLocation,
)
from .result import Result, Ok, Err

__all__ = [
    # Client
    "OpenAIWebSearchClient",
    "main",
    "cli",
    "web_search",
    # Types
    "SearchResponse",
    "UserLocation",
    "ReasoningEffort",
    "SearchType",
    "SearchContextSize",
    "ApiError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "0.5.0"
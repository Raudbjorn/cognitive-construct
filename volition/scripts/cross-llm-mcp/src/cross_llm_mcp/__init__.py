"""
cross-llm-mcp - Multi-provider LLM routing

Usage:
    from cross_llm_mcp import CrossLLMClient

    client = CrossLLMClient()

    # Auto-select provider based on available API keys
    result = await client.call("Explain recursion")
    if result.is_ok():
        print(result.value.response)
    else:
        print(f"Error: {result.error.message}")

    # Use specific provider
    result = await client.call("Write a haiku", provider="anthropic")

    # Use tag-based routing (coding, creative, general, math, etc.)
    result = await client.call("Fix this code", tag="coding")
"""

from .client import CrossLLMClient
from .types import (
    ApiError,
    LLMResponse,
    Provider,
    ProviderConfig,
)
from .result import Result, Ok, Err
from .cli import main

__all__ = [
    # Client
    "CrossLLMClient",
    "main",
    # Types
    "LLMResponse",
    "ProviderConfig",
    "Provider",
    "ApiError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "0.1.0"

"""
perplexity - Python client library for Perplexity AI API

Usage:
    from perplexity import PerplexityClient, ask, chat

    # Quick question
    result = await ask("What is the capital of France?")
    if result.is_ok():
        print(result.value)

    # Client-based usage
    client = PerplexityClient(api_key="your-key")  # or set PERPLEXITY_API_KEY

    # Simple question
    result = await client.ask("Explain quantum computing briefly.")
    if result.is_ok():
        print(result.value)

    # Full chat with messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the latest developments in AI?"},
    ]
    result = await client.chat(messages)
    if result.is_ok():
        print(result.value.content)
        print(f"Citations: {result.value.citations}")
"""

from .client import PerplexityClient, ask, chat
from .types import (
    AVAILABLE_MODELS,
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Model,
    PerplexityError,
    Usage,
)
from .result import Result, Ok, Err

__all__ = [
    "PerplexityClient",
    "ask",
    "chat",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Message",
    "Model",
    "Usage",
    "PerplexityError",
    "AVAILABLE_MODELS",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

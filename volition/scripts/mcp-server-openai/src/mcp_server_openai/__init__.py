"""
mcp-server-openai - OpenAI API client and MCP server

Usage:
    from mcp_server_openai import OpenAIClient

    client = OpenAIClient(api_key="...")  # or set OPENAI_API_KEY env var

    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4"
    )
    if result.is_ok():
        print(result.value.choices[0].message.content)
    else:
        print(f"Error: {result.error.message}")

MCP Server:
    from mcp_server_openai import run_stdio_server
    await run_stdio_server(openai_api_key="...")
"""

from .client import OpenAIClient
from .types import (
    ApiError,
    ChatCompletionResponse,
    ChatCompletionRequest,
    Choice,
    Message,
    Role,
)
from .result import Result, Ok, Err
from .server import run_stdio_server

__all__ = [
    # Client
    "OpenAIClient",
    "run_stdio_server",
    # Types
    "ChatCompletionResponse",
    "ChatCompletionRequest",
    "Choice",
    "Message",
    "Role",
    "ApiError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "0.1.0"
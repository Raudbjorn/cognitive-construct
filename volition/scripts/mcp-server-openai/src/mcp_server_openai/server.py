import asyncio
import logging

import mcp
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .client import OpenAIClient
from .types import Role

logger = logging.getLogger(__name__)


def build_server(*, openai_api_key: str, server_name: str = "openai-server") -> Server:
    server = Server(server_name)
    client = OpenAIClient(api_key=openai_api_key)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="ask-openai",
                description="Ask OpenAI models a direct question",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Prompt to send to OpenAI"},
                        "model": {
                            "type": "string",
                            "default": "gpt-4",
                            "enum": ["gpt-4", "gpt-3.5-turbo"],
                        },
                        "temperature": {"type": "number", "default": 0.7, "minimum": 0, "maximum": 2},
                        "max_tokens": {"type": "integer", "default": 500, "minimum": 1, "maximum": 4000},
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def handle_tool_call(name: str, arguments: dict | None) -> list[types.TextContent]:
        if name != "ask-openai":
            return [types.TextContent(type="text", text=f"Error: unknown tool: {name}")]
        if not arguments:
            return [types.TextContent(type="text", text="Error: no arguments provided")]
        if "query" not in arguments:
            return [types.TextContent(type="text", text="Error: missing required argument: query")]

        try:
            temperature = float(arguments.get("temperature", 0.7))
            max_tokens = int(arguments.get("max_tokens", 500))
        except (TypeError, ValueError):
            return [types.TextContent(type="text", text="Error: invalid numeric argument")]

        query = str(arguments["query"])
        model = str(arguments.get("model", "gpt-4"))

        result = await client.chat_completion(
            messages=[
                {"role": Role.SYSTEM, "content": "You are a helpful assistant."},
                {"role": Role.USER, "content": query},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if result.is_err():
            return [types.TextContent(type="text", text=f"Error: {result.error.message}")]
        
        response = result.value
        if not response.choices:
             return [types.TextContent(type="text", text="Error: OpenAI response was empty (no choices)")]
        
        content = response.choices[0].message.content
        if not content:
            return [types.TextContent(type="text", text="Error: OpenAI response was empty (no content)")]

        return [types.TextContent(type="text", text=f"OpenAI Response:\n{content}")]

    return server


async def run_stdio_server(*, openai_api_key: str, server_name: str = "openai-server", server_version: str = "0.1.0") -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        server = build_server(openai_api_key=openai_api_key, server_name=server_name)
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=server_name,
                server_version=server_version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
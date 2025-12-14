"""Cross LLM CLI implementation."""

import argparse
import asyncio
import json
import sys
from typing import Any

from .client import CrossLLMClient
from .config import PROVIDERS
from .types import LLMResponse


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    """Convert LLMResponse to dict for JSON output."""
    return {
        "provider": resp.provider,
        "model": resp.model,
        "response": resp.response,
        "usage": resp.usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-LLM - Multi-provider LLM routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m cross_llm_mcp call "Explain recursion" --provider openai
    python -m cross_llm_mcp call "Write a haiku" --tag creative
    python -m cross_llm_mcp call "Fix this code" --tag coding --json
    python -m cross_llm_mcp providers
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # call command
    call_parser = subparsers.add_parser("call", help="Call an LLM with a prompt")
    call_parser.add_argument("prompt", help="The prompt to send")
    call_parser.add_argument(
        "-p", "--provider",
        choices=list(PROVIDERS.keys()),
        help="Provider to use",
    )
    call_parser.add_argument("-m", "--model", help="Model to use")
    call_parser.add_argument(
        "-t", "--tag",
        default="general",
        help="Task tag for routing (coding, creative, general, etc.)",
    )
    call_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    # providers command
    subparsers.add_parser("providers", help="List available providers")

    # models command
    models_parser = subparsers.add_parser("models", help="List models for a provider")
    models_parser.add_argument("provider", choices=list(PROVIDERS.keys()), help="Provider name")

    args = parser.parse_args()
    client = CrossLLMClient()

    if args.command == "providers":
        available = client.get_available_providers()
        print("Available providers:")
        for p in PROVIDERS:
            status = "✓" if p in available else "✗"
            env_key = PROVIDERS[p].env_key
            print(f"  {status} {p} ({env_key})")

    elif args.command == "models":
        provider = args.provider
        print(f"Default model for {provider}: {PROVIDERS[provider].default_model}")

    elif args.command == "call":
        result = asyncio.run(client.call(
            prompt=args.prompt,
            provider=args.provider,
            model=args.model,
            tag=args.tag,
        ))

        if args.json:
            if result.is_ok():
                print(json.dumps({
                    "status": "success",
                    **_response_to_dict(result.value)
                }, indent=2))
            else:
                print(json.dumps({
                    "status": "error",
                    "error": result.error.message
                }, indent=2))
        else:
            if result.is_ok():
                print(result.value.response)
            else:
                print(f"Error: {result.error.message}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()

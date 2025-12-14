"""HTTP adapter factory and exports.

This module provides HTTP adapters for AI model providers:
- OllamaAdapter: Local Ollama API
- LMStudioAdapter: Local LM Studio API
- OpenRouterAdapter: OpenRouter API gateway
- OpenAIAdapter: OpenAI API
- AnthropicAdapter: Anthropic API
"""

from typing import Type

from adapters.anthropic import AnthropicAdapter
from adapters.base_http import BaseHTTPAdapter
from adapters.lmstudio import LMStudioAdapter
from adapters.ollama import OllamaAdapter
from adapters.openai import OpenAIAdapter
from adapters.openrouter import OpenRouterAdapter
from models.config import HTTPAdapterConfig


def create_adapter(
    name: str, config: HTTPAdapterConfig
) -> BaseHTTPAdapter:
    """Factory function to create HTTP adapter.

    Args:
        name: Adapter name ('ollama', 'lmstudio', 'openrouter', 'openai', 'anthropic')
        config: HTTP adapter configuration

    Returns:
        HTTP adapter instance

    Raises:
        ValueError: If adapter is not supported
    """
    http_adapters: dict[str, Type[BaseHTTPAdapter]] = {
        "ollama": OllamaAdapter,
        "lmstudio": LMStudioAdapter,
        "openrouter": OpenRouterAdapter,
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
    }

    if name not in http_adapters:
        supported = ", ".join(sorted(http_adapters.keys()))
        raise ValueError(
            f"Unknown adapter: '{name}'. Supported adapters: {supported}"
        )

    return http_adapters[name](
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        api_key=config.api_key,
        headers=config.headers,
    )


__all__ = [
    "AnthropicAdapter",
    "BaseHTTPAdapter",
    "LMStudioAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenRouterAdapter",
    "create_adapter",
]

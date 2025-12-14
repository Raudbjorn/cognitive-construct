"""Internal adapter factory for AI Counsel.

Re-exports the adapter factory from the main adapters module.
This provides a stable import path for the client library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ai_counsel._internal.config import HTTPAdapterConfig

# Add project root to path for importing existing adapters
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from adapters.anthropic import AnthropicAdapter
from adapters.base_http import BaseHTTPAdapter
from adapters.lmstudio import LMStudioAdapter
from adapters.ollama import OllamaAdapter
from adapters.openai import OpenAIAdapter
from adapters.openrouter import OpenRouterAdapter


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

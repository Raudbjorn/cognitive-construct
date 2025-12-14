"""Pytest fixtures for all test modules."""
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from adapters.base_http import BaseHTTPAdapter


class MockHTTPAdapter(BaseHTTPAdapter):
    """Mock HTTP adapter for testing."""

    def __init__(self, name: str, timeout: int = 60):
        """Initialize mock adapter."""
        super().__init__(base_url=f"http://mock-{name}:8000", timeout=timeout)
        self.name = name
        self.invoke_mock = AsyncMock()
        self.response_counter = 0
        self._set_default_responses()

    def _set_default_responses(self):
        """Set sensible default responses for mock deliberations."""
        default_response = "After careful analysis, I believe the proposed approach has merit. It addresses the core concerns while maintaining practical feasibility. The implementation timeline seems reasonable."
        self.invoke_mock.return_value = default_response
        self.response_counter = 0

    def build_request(self, model: str, prompt: str):
        """Mock build_request method."""
        return (
            "/api/generate",
            {"Content-Type": "application/json"},
            {"model": model, "prompt": prompt},
        )

    def parse_response(self, response_json: dict) -> str:
        """Mock parse_response method."""
        return response_json.get("response", "")

    async def invoke(
        self,
        prompt: str,
        model: str,
        context: Optional[str] = None,
        is_deliberation: bool = True,
    ) -> str:
        """Mock invoke method."""
        result = await self.invoke_mock(prompt, model, context, is_deliberation)
        self.response_counter += 1
        return result


@pytest.fixture
def mock_adapters():
    """
    Create mock adapters for testing deliberation engine.

    Returns:
        dict: Dictionary of mock adapters by name
    """
    ollama = MockHTTPAdapter("ollama")
    lmstudio = MockHTTPAdapter("lmstudio")
    openrouter = MockHTTPAdapter("openrouter")

    # Set default return values
    ollama.invoke_mock.return_value = "Ollama response"
    lmstudio.invoke_mock.return_value = "LMStudio response"
    openrouter.invoke_mock.return_value = "OpenRouter response"

    return {
        "ollama": ollama,
        "lmstudio": lmstudio,
        "openrouter": openrouter,
    }


@pytest.fixture
def sample_config():
    """
    Sample configuration for testing.

    Returns:
        dict: Sample configuration dict
    """
    return {
        "defaults": {
            "mode": "quick",
            "rounds": 2,
            "max_rounds": 5,
            "timeout_per_round": 60,
        },
        "storage": {
            "transcripts_dir": "transcripts",
            "format": "markdown",
            "auto_export": True,
        },
        "deliberation": {
            "convergence_threshold": 0.8,
            "enable_convergence_detection": True,
        },
    }

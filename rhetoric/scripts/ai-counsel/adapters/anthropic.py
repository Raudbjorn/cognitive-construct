"""Anthropic HTTP adapter for AI Counsel."""

from typing import Optional, Tuple

from adapters.base_http import BaseHTTPAdapter


class AnthropicAdapter(BaseHTTPAdapter):
    """HTTP adapter for Anthropic API.

    Example:
        adapter = AnthropicAdapter(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-...",
        )
        result = await adapter.invoke(prompt="Hello", model="claude-3-5-sonnet-20241022")
    """

    ANTHROPIC_VERSION = "2023-06-01"

    def build_request(
        self, model: str, prompt: str
    ) -> Tuple[str, dict[str, str], dict]:
        """Build Anthropic messages request.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            prompt: The prompt to send

        Returns:
            Tuple of (endpoint, headers, body)
        """
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.ANTHROPIC_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # Merge default headers
        headers.update(self.default_headers)

        body = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        return "/v1/messages", headers, body

    def parse_response(self, response_json: dict) -> str:
        """Parse Anthropic response to extract model output.

        Args:
            response_json: Parsed JSON response from API

        Returns:
            Extracted model response text
        """
        content = response_json.get("content", [])
        if not content:
            return ""

        # Find text block
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")

        # Fallback to first content block
        if content and "text" in content[0]:
            return content[0]["text"]

        return ""

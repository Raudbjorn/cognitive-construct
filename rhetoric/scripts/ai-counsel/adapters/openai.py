"""OpenAI HTTP adapter for AI Counsel."""

from typing import Optional, Tuple

from adapters.base_http import BaseHTTPAdapter


class OpenAIAdapter(BaseHTTPAdapter):
    """HTTP adapter for OpenAI API.

    Example:
        adapter = OpenAIAdapter(
            base_url="https://api.openai.com",
            api_key="sk-...",
        )
        result = await adapter.invoke(prompt="Hello", model="gpt-4o")
    """

    def build_request(
        self, model: str, prompt: str
    ) -> Tuple[str, dict[str, str], dict]:
        """Build OpenAI chat completions request.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
            prompt: The prompt to send

        Returns:
            Tuple of (endpoint, headers, body)
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Merge default headers
        headers.update(self.default_headers)

        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        }

        return "/v1/chat/completions", headers, body

    def parse_response(self, response_json: dict) -> str:
        """Parse OpenAI response to extract model output.

        Args:
            response_json: Parsed JSON response from API

        Returns:
            Extracted model response text
        """
        choices = response_json.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "")

"""OpenAI API client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .result import Err, Ok, Result
from .types import (
    ApiError,
    ChatCompletionResponse,
    Choice,
    Message,
    Role,
)

# === Constants ===

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0
API_KEY_ENV_VAR = "OPENAI_API_KEY"


# === Client Class ===

@dataclass
class OpenAIClient:
    """Client for OpenAI API.

    Usage:
        client = OpenAIClient(api_key="...")

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4"
        )
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize resolved API key."""
        self._resolved_key = self.api_key or os.environ.get(API_KEY_ENV_VAR)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self._resolved_key}",
            "Content-Type": "application/json",
        }

    def _check_api_key(self) -> Result[None, ApiError]:
        """Verify API key is configured."""
        if not self._resolved_key:
            return Err(
                ApiError(
                    message=f"API key required. Set {API_KEY_ENV_VAR} or pass api_key parameter.",
                    code="AUTH_REQUIRED",
                )
            )
        return Ok(None)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Result[dict[str, Any], ApiError]:
        """Make HTTP request with error handling."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=self._headers(),
                )

                if not response.is_success:
                    try:
                        error_data = response.json()
                        # OpenAI error format often has {"error": {"message": ...}}
                        if "error" in error_data and isinstance(error_data["error"], dict):
                            message = error_data["error"].get("message")
                            code = error_data["error"].get("code")
                        else:
                            message = error_data.get("error") or response.reason_phrase
                            code = None
                    except Exception:
                        message = response.reason_phrase or f"HTTP {response.status_code}"
                        code = None

                    return Err(
                        ApiError(
                            message=str(message),
                            code=str(code) if code else None,
                            status_code=response.status_code,
                        )
                    )

                return Ok(response.json())

        except httpx.TimeoutException:
            return Err(ApiError(message="Request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(ApiError(message=f"Request failed: {e}", code="REQUEST_ERROR"))

    # === Public Methods ===

    async def chat_completion(
        self,
        messages: list[Message] | list[dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Result[ChatCompletionResponse, ApiError]:
        """Send a chat completion request."""
        
        # Convert dicts to Message objects if necessary
        fmt_messages = []
        for m in messages:
            if isinstance(m, dict):
                fmt_messages.append({"role": m["role"], "content": m["content"]})
            else:
                fmt_messages.append({"role": m.role.value if isinstance(m.role, Role) else m.role, "content": m.content})

        payload = {
            "model": model,
            "messages": fmt_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        result = await self._request("POST", "/chat/completions", json_body=payload)

        if result.is_err():
            return result  # type: ignore

        data = result.value
        
        # Parse choices
        choices = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", "")
                ),
                finish_reason=c.get("finish_reason")
            ))

        return Ok(
            ChatCompletionResponse(
                id=data.get("id", ""),
                object=data.get("object", ""),
                created=data.get("created", 0),
                model=data.get("model", ""),
                choices=choices,
                usage=data.get("usage")
            )
        )

"""Perplexity AI API client library."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .result import Err, Ok, Result
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

PERPLEXITY_API_BASE = "https://api.perplexity.ai"
DEFAULT_TIMEOUT = 120.0


def _get_api_key(api_key: str | None = None) -> str | None:
    """Get API key from parameter or environment."""
    return api_key or os.environ.get("PERPLEXITY_API_KEY")


def _parse_response(data: dict[str, Any]) -> ChatResponse:
    """Parse chat completion response."""
    choices = []
    for c in data.get("choices", []):
        msg = c.get("message", {})
        choices.append(
            Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            )
        )

    usage = None
    if "usage" in data:
        u = data["usage"]
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

    return ChatResponse(
        id=data.get("id", ""),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
        citations=data.get("citations", []),
    )


async def chat(
    messages: list[dict[str, str]] | list[Message],
    model: str | Model = Model.SONAR_SMALL,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result[ChatResponse, PerplexityError]:
    """Send a chat completion request to Perplexity API.

    Args:
        messages: List of messages (dicts with 'role' and 'content', or Message objects)
        model: Model to use (default: sonar-small)
        api_key: API key (or set PERPLEXITY_API_KEY env var)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-2)
        timeout: Request timeout in seconds

    Returns:
        Result containing ChatResponse on success or PerplexityError on failure
    """
    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        return Err(
            PerplexityError("Missing PERPLEXITY_API_KEY (set env var or pass api_key parameter)")
        )

    # Normalize model
    model_str = model.value if isinstance(model, Model) else model

    # Normalize messages
    msg_list = []
    for m in messages:
        if isinstance(m, Message):
            msg_list.append({"role": m.role, "content": m.content})
        else:
            msg_list.append(m)

    payload: dict[str, Any] = {
        "model": model_str,
        "messages": msg_list,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{PERPLEXITY_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        return Ok(_parse_response(data))

    except httpx.TimeoutException:
        return Err(PerplexityError("Request timed out"))
    except httpx.HTTPStatusError as e:
        return Err(
            PerplexityError(f"HTTP error: {e.response.status_code}", e.response.status_code)
        )
    except httpx.RequestError as e:
        return Err(PerplexityError(f"Request failed: {e}"))
    except json.JSONDecodeError:
        return Err(PerplexityError("Invalid JSON response"))


async def ask(
    question: str,
    system_prompt: str | None = None,
    model: str | Model = Model.SONAR_SMALL,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Result[str, PerplexityError]:
    """Ask a single question and get the response text.

    Convenience function for simple Q&A usage.

    Args:
        question: The question to ask
        system_prompt: Optional system prompt
        model: Model to use
        api_key: API key
        timeout: Request timeout

    Returns:
        Result containing response text on success or PerplexityError on failure
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    result = await chat(messages, model, api_key, timeout=timeout)
    if result.is_err():
        return result

    return Ok(result.value.content)


@dataclass
class PerplexityClient:
    """Perplexity AI API client.

    Usage:
        client = PerplexityClient(api_key="your-key")  # or set PERPLEXITY_API_KEY

        # Simple question
        result = await client.ask("What is the capital of France?")
        if result.is_ok():
            print(result.value)

        # Full chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."},
        ]
        result = await client.chat(messages)
        if result.is_ok():
            print(result.value.content)
            print(f"Citations: {result.value.citations}")
    """

    api_key: str | None = None
    model: str | Model = Model.SONAR_SMALL
    timeout: float = DEFAULT_TIMEOUT

    async def chat(
        self,
        messages: list[dict[str, str]] | list[Message],
        model: str | Model | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> Result[ChatResponse, PerplexityError]:
        """Send a chat completion request."""
        return await chat(
            messages=messages,
            model=model or self.model,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout,
        )

    async def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        model: str | Model | None = None,
    ) -> Result[str, PerplexityError]:
        """Ask a single question."""
        return await ask(
            question=question,
            system_prompt=system_prompt,
            model=model or self.model,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @staticmethod
    def available_models() -> list[str]:
        """Get list of available models."""
        return AVAILABLE_MODELS

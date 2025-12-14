"""LLM provider implementations for vibe check."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from .config import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_BASE_URL_ENV,
    ANTHROPIC_DEFAULT_BASE_URL,
    ANTHROPIC_DEFAULT_VERSION,
    ANTHROPIC_VERSION_ENV,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    FALLBACK_GEMINI_MODEL,
    GEMINI_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
)
from .result import Err, Ok, Result
from .types import VibeCheckError


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Response from an LLM provider."""

    text: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Result[LLMResponse, VibeCheckError]:
        """Generate a response from the LLM."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
        self._client: httpx.AsyncClient | None = None

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Result[LLMResponse, VibeCheckError]:
        if not self._api_key:
            return Err(VibeCheckError(message="Gemini API key missing", code="AUTH_REQUIRED"))

        model_name = model or DEFAULT_GEMINI_MODEL
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Try primary model, fall back to secondary
        for try_model in [model_name, FALLBACK_GEMINI_MODEL]:
            result = await self._call_gemini(full_prompt, try_model)
            if result.is_ok():
                return result

            # Only retry if it was the primary model that failed
            if try_model == FALLBACK_GEMINI_MODEL:
                return result

        return Err(VibeCheckError(message="Gemini request failed", code="PROVIDER_ERROR"))

    async def _call_gemini(
        self,
        prompt: str,
        model: str,
    ) -> Result[LLMResponse, VibeCheckError]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    params={"key": self._api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    headers={"Content-Type": "application/json"},
                )

                if not response.is_success:
                    return Err(
                        VibeCheckError(
                            message=f"Gemini request failed: {response.status_code}",
                            code="PROVIDER_ERROR",
                            status_code=response.status_code,
                        )
                    )

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return Err(VibeCheckError(message="No response from Gemini", code="EMPTY_RESPONSE"))

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    return Err(VibeCheckError(message="No content from Gemini", code="EMPTY_RESPONSE"))

                text = parts[0].get("text", "")
                return Ok(LLMResponse(text=text))

        except httpx.TimeoutException:
            return Err(VibeCheckError(message="Gemini request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(VibeCheckError(message=f"Gemini request failed: {e}", code="REQUEST_ERROR"))


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Result[LLMResponse, VibeCheckError]:
        if not self._api_key:
            return Err(VibeCheckError(message="OpenAI API key missing", code="AUTH_REQUIRED"))

        model_name = model or DEFAULT_OPENAI_MODEL
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [{"role": "system", "content": full_prompt}],
                    },
                )

                if not response.is_success:
                    return Err(
                        VibeCheckError(
                            message=f"OpenAI request failed: {response.status_code}",
                            code="PROVIDER_ERROR",
                            status_code=response.status_code,
                        )
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return Err(VibeCheckError(message="No response from OpenAI", code="EMPTY_RESPONSE"))

                message = choices[0].get("message", {})
                text = message.get("content", "")
                return Ok(LLMResponse(text=text))

        except httpx.TimeoutException:
            return Err(VibeCheckError(message="OpenAI request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(VibeCheckError(message=f"OpenAI request failed: {e}", code="REQUEST_ERROR"))


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Result[LLMResponse, VibeCheckError]:
        if not self._api_key:
            return Err(VibeCheckError(message="OpenRouter API key missing", code="AUTH_REQUIRED"))

        if not model:
            return Err(
                VibeCheckError(
                    message="OpenRouter requires a model to be specified",
                    code="MODEL_REQUIRED",
                )
            )

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Vibe Check Client",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "system", "content": full_prompt}],
                    },
                )

                if not response.is_success:
                    return Err(
                        VibeCheckError(
                            message=f"OpenRouter request failed: {response.status_code}",
                            code="PROVIDER_ERROR",
                            status_code=response.status_code,
                        )
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return Err(VibeCheckError(message="No response from OpenRouter", code="EMPTY_RESPONSE"))

                message = choices[0].get("message", {})
                text = message.get("content", "")
                return Ok(LLMResponse(text=text))

        except httpx.TimeoutException:
            return Err(VibeCheckError(message="OpenRouter request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(VibeCheckError(message=f"OpenRouter request failed: {e}", code="REQUEST_ERROR"))


class AnthropicProvider(LLMProvider):
    """Anthropic provider."""

    def __init__(
        self,
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
    ):
        self._api_key = api_key or os.environ.get(ANTHROPIC_API_KEY_ENV)
        self._auth_token = auth_token or os.environ.get(ANTHROPIC_AUTH_TOKEN_ENV)
        self._base_url = (base_url or os.environ.get(ANTHROPIC_BASE_URL_ENV) or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        self._version = os.environ.get(ANTHROPIC_VERSION_ENV, ANTHROPIC_DEFAULT_VERSION)

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._version,
        }

        if self._api_key:
            headers["x-api-key"] = self._api_key
        elif self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        return headers

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Result[LLMResponse, VibeCheckError]:
        if not self._api_key and not self._auth_token:
            return Err(
                VibeCheckError(
                    message="Anthropic API key or auth token required",
                    code="AUTH_REQUIRED",
                )
            )

        model_name = model or DEFAULT_ANTHROPIC_MODEL

        body: dict = {
            "model": model_name,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            body["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._base_url}/v1/messages",
                    headers=self._build_headers(),
                    json=body,
                )

                raw_text = response.text
                parsed_body = None
                if raw_text:
                    try:
                        parsed_body = response.json()
                    except Exception:
                        pass

                if not response.is_success:
                    request_id = (
                        response.headers.get("anthropic-request-id")
                        or response.headers.get("x-request-id")
                    )
                    request_suffix = f" (request id: {request_id})" if request_id else ""

                    if parsed_body:
                        error_msg = (
                            parsed_body.get("error", {}).get("message")
                            or parsed_body.get("message")
                            or raw_text.strip()
                        )
                    else:
                        error_msg = raw_text.strip() if raw_text else "Unknown error"

                    if response.status_code in (401, 403):
                        return Err(
                            VibeCheckError(
                                message=f"Anthropic authentication failed{request_suffix}",
                                code="AUTH_ERROR",
                                status_code=response.status_code,
                            )
                        )

                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        retry_msg = f" Retry after {retry_after}s." if retry_after else ""
                        return Err(
                            VibeCheckError(
                                message=f"Anthropic rate limit exceeded{request_suffix}.{retry_msg}",
                                code="RATE_LIMITED",
                                status_code=429,
                            )
                        )

                    return Err(
                        VibeCheckError(
                            message=f"Anthropic request failed: {error_msg}{request_suffix}",
                            code="PROVIDER_ERROR",
                            status_code=response.status_code,
                        )
                    )

                content = parsed_body.get("content", []) if parsed_body else []
                text_block = next(
                    (block for block in content if block.get("type") == "text"),
                    None,
                )

                if text_block:
                    return Ok(LLMResponse(text=text_block.get("text", "")))

                # Fallback to first content block
                if content and content[0].get("text"):
                    return Ok(LLMResponse(text=content[0]["text"]))

                return Ok(LLMResponse(text=""))

        except httpx.TimeoutException:
            return Err(VibeCheckError(message="Anthropic request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(VibeCheckError(message=f"Anthropic request failed: {e}", code="REQUEST_ERROR"))


def get_provider(provider_name: str) -> LLMProvider:
    """Get an LLM provider by name."""
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")

    return provider_class()

"""Cross LLM client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import PROVIDERS, TAG_ROUTING
from .result import Err, Ok, Result
from .types import ApiError, LLMResponse, Provider


@dataclass
class CrossLLMClient:
    """Client for calling multiple LLM providers."""

    timeout: float = 60.0

    def get_available_providers(self) -> list[Provider]:
        """Get list of providers with API keys configured."""
        return [
            p for p, config in PROVIDERS.items() 
            if os.environ.get(config.env_key)
        ] # type: ignore

    def select_provider(self, tag: str = "general", preferred: Provider | None = None) -> Provider | None:
        """Select best available provider based on tag and preference."""
        available = self.get_available_providers()
        if not available:
            return None

        # Use preferred if available
        if preferred and preferred in available:
            return preferred

        # Use tag-based routing
        preferences = TAG_ROUTING.get(tag, TAG_ROUTING["general"])
        for p in preferences:
            if p in available:
                return p # type: ignore

        # Fallback to first available
        return available[0]

    async def _call_openai(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["openai"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )
                
                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="openai", status_code=response.status_code))

                data = response.json()
                return Ok(LLMResponse(
                    provider="openai",
                    model=model,
                    response=data["choices"][0]["message"]["content"],
                    usage=data.get("usage"),
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="openai"))

    async def _call_anthropic(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["anthropic"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{config.base_url}/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                
                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="anthropic", status_code=response.status_code))
                
                data = response.json()
                return Ok(LLMResponse(
                    provider="anthropic",
                    model=model,
                    response=data["content"][0]["text"],
                    usage=data.get("usage"),
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="anthropic"))

    async def _call_deepseek(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["deepseek"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )

                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="deepseek", status_code=response.status_code))

                data = response.json()
                return Ok(LLMResponse(
                    provider="deepseek",
                    model=model,
                    response=data["choices"][0]["message"]["content"],
                    usage=data.get("usage"),
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="deepseek"))

    async def _call_gemini(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["gemini"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{config.base_url}/models/{model}:generateContent?key={api_key}"
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )

                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="gemini", status_code=response.status_code))

                data = response.json()
                return Ok(LLMResponse(
                    provider="gemini",
                    model=model,
                    response=data["candidates"][0]["content"]["parts"][0]["text"],
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="gemini"))

    async def _call_grok(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["grok"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )

                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="grok", status_code=response.status_code))

                data = response.json()
                return Ok(LLMResponse(
                    provider="grok",
                    model=model,
                    response=data["choices"][0]["message"]["content"],
                    usage=data.get("usage"),
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="grok"))

    async def _call_mistral(self, prompt: str, model: str, api_key: str) -> Result[LLMResponse, ApiError]:
        config = PROVIDERS["mistral"]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )

                if not response.is_success:
                     return Err(ApiError(message=response.text, provider="mistral", status_code=response.status_code))

                data = response.json()
                return Ok(LLMResponse(
                    provider="mistral",
                    model=model,
                    response=data["choices"][0]["message"]["content"],
                    usage=data.get("usage"),
                ))
        except Exception as e:
            return Err(ApiError(message=str(e), provider="mistral"))

    async def call(
        self,
        prompt: str,
        provider: Provider | None = None,
        model: str | None = None,
        tag: str = "general",
    ) -> Result[LLMResponse, ApiError]:
        """Call an LLM with automatic provider selection."""
        selected_provider = self.select_provider(tag, provider)
        
        if not selected_provider:
             return Err(ApiError(message="No API keys configured or no provider available.", provider=provider or "openai"))

        config = PROVIDERS[selected_provider]
        api_key = os.environ.get(config.env_key)
        
        if not api_key:
             return Err(ApiError(message=f"API key not found for {selected_provider}. Set {config.env_key}", provider=selected_provider))

        selected_model = model or config.default_model

        method_name = f"_call_{selected_provider}"
        if not hasattr(self, method_name):
             return Err(ApiError(message=f"Provider implementation not found: {selected_provider}", provider=selected_provider))
        
        caller = getattr(self, method_name)
        return await caller(prompt, selected_model, api_key)

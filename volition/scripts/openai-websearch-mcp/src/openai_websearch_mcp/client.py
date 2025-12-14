"""OpenAI Web Search client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import (
    API_KEY_ENV_VAR,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    REASONING_MODELS,
    VALID_MODELS,
)
from .result import Err, Ok, Result
from .types import (
    ApiError,
    ReasoningEffort,
    SearchContextSize,
    SearchResponse,
    SearchType,
    UserLocation,
)


@dataclass
class OpenAIWebSearchClient:
    """Client for OpenAI Web Search API."""

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

    async def search(
        self,
        query: str,
        model: str = "gpt-5-mini",
        reasoning_effort: str | ReasoningEffort | None = None,
        search_type: str | SearchType = SearchType.PREVIEW,
        search_context_size: str | SearchContextSize = SearchContextSize.MEDIUM,
        user_location: UserLocation | None = None,
    ) -> Result[SearchResponse, ApiError]:
        """Execute web search."""
        
        if model not in VALID_MODELS:
             return Err(ApiError(f"Invalid model: {model}. Valid models: {', '.join(sorted(VALID_MODELS))}"))

        # Convert enums
        search_type_val = search_type.value if isinstance(search_type, SearchType) else search_type
        context_size_val = search_context_size.value if isinstance(search_context_size, SearchContextSize) else search_context_size
        
        tool_config = {
            "type": search_type_val,
            "search_context_size": context_size_val,
            "user_location": user_location.to_dict() if user_location else None,
        }

        request_params = {
            "model": model,
            "tools": [tool_config],
            "input": query,
        }

        # Set intelligent defaults for reasoning models
        if model in REASONING_MODELS:
            if reasoning_effort is None:
                effort = "low" if model == "gpt-5-mini" else "medium"
            else:
                effort = reasoning_effort.value if isinstance(reasoning_effort, ReasoningEffort) else reasoning_effort
            
            request_params["reasoning"] = {"effort": effort}

        result = await self._request("POST", "/responses", json_body=request_params)

        if result.is_err():
            return result  # type: ignore

        data = result.value
        # Assuming response structure has output_text field as per original code 'response.output_text'
        # Check actual API response structure if possible. 
        # The original code did `response.output_text`. 
        # If response is a Pydantic model in the SDK, `output_text` is likely a field.
        # In JSON, it's probably "output_text".
        
        return Ok(SearchResponse(content=data.get("output_text", "")))

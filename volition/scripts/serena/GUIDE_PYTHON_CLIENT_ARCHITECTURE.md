# Python Client Library Architecture Guide

A systematic approach for creating high-quality, consistent Python client libraries for API services.

## Design Philosophy

1. **Errors as values, not exceptions** - Use `Result[T, E]` types
2. **Immutable by default** - Frozen dataclasses with slots
3. **Async-first** - httpx for HTTP, asyncio for concurrency
4. **Type-safe** - Full type hints, mypy strict mode
5. **Zero surprises** - Explicit configuration, no magic
6. **Minimal dependencies** - httpx for HTTP, stdlib for everything else

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        __init__.py                          │
│              (Public API, exports, version)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         client.py                           │
│         (Client class, convenience functions)               │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Config     │    │   Methods    │    │  Convenience │  │
│  │  (api_key,   │───▶│  (async ops) │◀───│  Functions   │  │
│  │   timeout)   │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    types.py     │  │    result.py    │  │   config.py     │
│   (Dataclasses, │  │   (Ok, Err,     │  │   (Constants,   │
│     Enums)      │  │    Result)      │  │    Defaults)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Module Specifications

### 1. result.py - Error Handling

The foundation of the error-as-values pattern:

```python
"""Result type for error handling as values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success case containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Get the value. Safe to call after checking is_ok()."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error case containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        """Raises ValueError. Check is_ok() first."""
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default since this is an error."""
        return default

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform success value (no-op for Err)."""
        return self  # type: ignore

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error."""
        return Err(fn(self.error))


# Type alias for the union
Result = Union[Ok[T], Err[E]]
```

**Usage patterns:**

```python
# Returning results
def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

# Consuming results
result = divide(10, 2)
if result.is_ok():
    print(f"Result: {result.value}")
else:
    print(f"Error: {result.error}")

# With unwrap_or
value = divide(10, 0).unwrap_or(0.0)  # Returns 0.0

# Chaining with map
result = divide(10, 2).map(lambda x: x * 2)  # Ok(10.0)
```

---

### 2. types.py - Data Types

All request/response types as frozen dataclasses:

```python
"""Type definitions for the API client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# === Enums ===

class OperationMode(str, Enum):
    """Operation mode options."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


class Status(str, Enum):
    """Status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# === Request Types ===

@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Search operation request."""

    query: str
    limit: int = 10
    mode: OperationMode = OperationMode.BALANCED
    filters: dict[str, Any] = field(default_factory=dict)


# === Response Types ===

@dataclass(frozen=True, slots=True)
class SearchResult:
    """Individual search result."""

    id: str
    title: str
    score: float
    url: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Search operation response."""

    query: str
    results: list[SearchResult]
    total_count: int
    has_more: bool = False


@dataclass(frozen=True, slots=True)
class Pagination:
    """Pagination information."""

    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool


# === Error Types ===

@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
```

**Design rules for types:**

1. Always use `frozen=True, slots=True`
2. Required fields before optional fields
3. Use `field(default_factory=...)` for mutable defaults
4. Prefer `str | None` over `Optional[str]`
5. Use Enum for fixed string choices
6. Include error types in the same module

---

### 3. client.py - Main Client

The client class and convenience functions:

```python
"""API client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .result import Err, Ok, Result
from .types import (
    ApiError,
    OperationMode,
    SearchResponse,
    SearchResult,
)

# === Constants ===

DEFAULT_BASE_URL = "https://api.service.com/v1"
DEFAULT_TIMEOUT = 30.0
API_KEY_ENV_VAR = "SERVICE_API_KEY"


# === Helper Functions ===

def _get_api_key(api_key: str | None = None) -> str | None:
    """Resolve API key from parameter or environment."""
    return api_key or os.environ.get(API_KEY_ENV_VAR)


def _parse_search_result(data: dict[str, Any]) -> SearchResult:
    """Parse API response into SearchResult."""
    return SearchResult(
        id=data.get("id", ""),
        title=data.get("title", ""),
        score=float(data.get("score", 0.0)),
        url=data.get("url"),
        snippet=data.get("snippet"),
        metadata=data.get("metadata", {}),
    )


# === Client Class ===

@dataclass
class ServiceClient:
    """Client for Service API.

    Usage:
        client = ServiceClient(api_key="...")  # or set SERVICE_API_KEY env var

        # Search
        result = await client.search("query")
        if result.is_ok():
            for item in result.value.results:
                print(f"{item.title}: {item.score}")

        # With options
        result = await client.search(
            "query",
            limit=20,
            mode=OperationMode.THOROUGH,
        )
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize resolved API key."""
        self._resolved_key = _get_api_key(self.api_key)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self._resolved_key}",
            "Content-Type": "application/json",
            "User-Agent": "serviceclient/1.0.0",
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
                        message = error_data.get("error") or error_data.get("message") or response.reason_phrase
                    except Exception:
                        message = response.reason_phrase or f"HTTP {response.status_code}"

                    return Err(
                        ApiError(
                            message=message,
                            status_code=response.status_code,
                        )
                    )

                return Ok(response.json())

        except httpx.TimeoutException:
            return Err(ApiError(message="Request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(ApiError(message=f"Request failed: {e}", code="REQUEST_ERROR"))

    # === Public Methods ===

    async def search(
        self,
        query: str,
        limit: int = 10,
        mode: OperationMode | str = OperationMode.BALANCED,
    ) -> Result[SearchResponse, ApiError]:
        """Search for items.

        Args:
            query: Search query string
            limit: Maximum results to return (default: 10)
            mode: Search mode (default: balanced)

        Returns:
            Result containing SearchResponse on success or ApiError on failure
        """
        # Normalize enum
        mode_str = mode.value if isinstance(mode, OperationMode) else mode

        result = await self._request(
            method="GET",
            path="/search",
            params={
                "q": query,
                "limit": limit,
                "mode": mode_str,
            },
        )

        if result.is_err():
            return result  # type: ignore

        data = result.value
        results = [_parse_search_result(r) for r in data.get("results", [])]

        return Ok(
            SearchResponse(
                query=query,
                results=results,
                total_count=data.get("total", len(results)),
                has_more=data.get("has_more", False),
            )
        )


# === Convenience Functions ===

async def search(
    query: str,
    limit: int = 10,
    api_key: str | None = None,
) -> Result[SearchResponse, ApiError]:
    """Search for items (convenience function).

    Creates a client for single use. For multiple requests,
    prefer creating a ServiceClient instance.
    """
    client = ServiceClient(api_key=api_key)
    return await client.search(query, limit)
```

**Client design rules:**

1. Use `@dataclass` (not `@dataclass(frozen=True)`) for clients - they have mutable state
2. Use `field(init=False, repr=False)` for internal state
3. Use `__post_init__` for initialization logic
4. Private methods start with `_`
5. All public methods return `Result[T, E]`
6. Include docstrings with usage examples
7. Provide convenience functions for one-off usage

---

### 4. __init__.py - Public API

```python
"""
serviceclient - Python client library for Service API

Usage:
    from serviceclient import ServiceClient, search

    # Client-based usage (recommended for multiple calls)
    client = ServiceClient(api_key="...")  # or set SERVICE_API_KEY env var

    result = await client.search("query")
    if result.is_ok():
        for item in result.value.results:
            print(f"{item.title}: {item.score}")
    else:
        print(f"Error: {result.error.message}")

    # Convenience function (for one-off calls)
    result = await search("query", limit=5)
"""

from .client import ServiceClient, search
from .types import (
    ApiError,
    OperationMode,
    Pagination,
    SearchResponse,
    SearchResult,
    Status,
)
from .result import Result, Ok, Err

__all__ = [
    # Client
    "ServiceClient",
    "search",
    # Types
    "SearchResponse",
    "SearchResult",
    "Pagination",
    "OperationMode",
    "Status",
    "ApiError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"
```

**Export rules:**

1. Client class first
2. Convenience functions second
3. Response types third
4. Request types fourth
5. Enums fifth
6. Error types sixth
7. Result types last
8. Always define `__all__`
9. Always define `__version__`

---

### 5. pyproject.toml - Package Config

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "serviceclient"
version = "1.0.0"
description = "Python client library for Service API"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Your Name" }]
keywords = ["service", "api", "client"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Typing :: Typed",
]
dependencies = [
    "httpx>=0.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "respx>=0.20",  # For mocking httpx
    "mypy>=1.0",
    "ruff>=0.1",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "W",    # pycodestyle warnings
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
]
```

---

## HTTP Patterns

### Retry with Exponential Backoff

```python
import asyncio
import math

async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> Result[httpx.Response, ApiError]:
    """Make request with exponential backoff retry."""
    last_error: ApiError | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return Ok(response)
        except httpx.HTTPStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.response.status_code < 500:
                return Err(ApiError(f"HTTP {e.response.status_code}", status_code=e.response.status_code))
            last_error = ApiError(f"HTTP {e.response.status_code}", status_code=e.response.status_code)
        except httpx.RequestError as e:
            last_error = ApiError(f"Request failed: {e}")

        if attempt < max_retries:
            delay = math.exp(attempt) * 0.05  # 50ms, 135ms, 370ms
            await asyncio.sleep(delay)

    return Err(last_error or ApiError("Request failed after retries"))
```

### Pagination Handling

```python
async def fetch_all_pages(
    client: ServiceClient,
    query: str,
    page_size: int = 100,
) -> Result[list[SearchResult], ApiError]:
    """Fetch all pages of results."""
    all_results: list[SearchResult] = []
    page = 1

    while True:
        result = await client.search(query, limit=page_size, page=page)
        if result.is_err():
            return result  # type: ignore

        all_results.extend(result.value.results)

        if not result.value.has_more:
            break

        page += 1

    return Ok(all_results)
```

### Concurrent Requests

```python
import asyncio

async def fetch_multiple(
    client: ServiceClient,
    queries: list[str],
    max_concurrent: int = 5,
) -> list[Result[SearchResponse, ApiError]]:
    """Fetch multiple queries with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(query: str) -> Result[SearchResponse, ApiError]:
        async with semaphore:
            return await client.search(query)

    tasks = [fetch_one(q) for q in queries]
    return await asyncio.gather(*tasks)
```

---

## Testing Patterns

### Basic Test Structure

```python
# tests/test_client.py
import pytest
import respx
from httpx import Response

from serviceclient import ServiceClient, ApiError

@pytest.fixture
def client():
    return ServiceClient(api_key="test-key")

@pytest.mark.asyncio
async def test_search_success(client):
    with respx.mock:
        respx.get("https://api.service.com/v1/search").mock(
            return_value=Response(200, json={
                "results": [{"id": "1", "title": "Test", "score": 0.9}],
                "total": 1,
            })
        )

        result = await client.search("test query")

        assert result.is_ok()
        assert len(result.value.results) == 1
        assert result.value.results[0].title == "Test"

@pytest.mark.asyncio
async def test_search_auth_error(client):
    with respx.mock:
        respx.get("https://api.service.com/v1/search").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        result = await client.search("test query")

        assert result.is_err()
        assert result.error.status_code == 401

@pytest.mark.asyncio
async def test_search_no_api_key():
    client = ServiceClient(api_key=None)  # No env var set

    result = await client.search("test query")

    assert result.is_err()
    assert "API key required" in result.error.message
```

---

## Quality Checklist

### Before Release

- [ ] All public functions have docstrings
- [ ] All public functions return `Result[T, E]`
- [ ] All dataclasses use `frozen=True, slots=True`
- [ ] No mutable default arguments
- [ ] Type hints on all functions
- [ ] `__all__` defined in `__init__.py`
- [ ] `__version__` defined in `__init__.py`
- [ ] `py.typed` marker file exists
- [ ] `pyproject.toml` has all metadata
- [ ] mypy passes with `strict = true`
- [ ] ruff passes with no errors
- [ ] Tests cover happy path and error cases
- [ ] README.md with installation and usage examples

### Code Style

- [ ] Line length <= 100 characters
- [ ] Imports sorted (stdlib, third-party, local)
- [ ] No unused imports
- [ ] No unused variables
- [ ] Consistent naming (snake_case for functions/variables, PascalCase for classes)
- [ ] Private functions/methods prefixed with `_`
- [ ] Constants in UPPER_CASE

---

## Anti-Patterns to Avoid

### 1. Exceptions for Expected Errors

```python
# BAD
async def fetch(url: str) -> dict:
    response = await client.get(url)
    if response.status_code == 404:
        raise NotFoundError()  # Don't raise for expected cases
    return response.json()

# GOOD
async def fetch(url: str) -> Result[dict, ApiError]:
    response = await client.get(url)
    if response.status_code == 404:
        return Err(ApiError("Not found", status_code=404))
    return Ok(response.json())
```

### 2. Mutable Default Arguments

```python
# BAD
@dataclass
class Request:
    tags: list[str] = []  # Shared across all instances!

# GOOD
@dataclass
class Request:
    tags: list[str] = field(default_factory=list)
```

### 3. Missing Timeouts

```python
# BAD
async with httpx.AsyncClient() as client:
    response = await client.get(url)  # Could hang forever

# GOOD
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
```

### 4. Swallowing Errors

```python
# BAD
try:
    return await client.get(url)
except Exception:
    return None  # What went wrong?

# GOOD
try:
    return Ok(await client.get(url))
except httpx.TimeoutException:
    return Err(ApiError("Request timed out"))
except httpx.RequestError as e:
    return Err(ApiError(f"Request failed: {e}"))
```

### 5. Inconsistent Return Types

```python
# BAD
async def search(query: str):  # Returns what? Could raise what?
    ...

# GOOD
async def search(query: str) -> Result[SearchResponse, ApiError]:
    ...
```

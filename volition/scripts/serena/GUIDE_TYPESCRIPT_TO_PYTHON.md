# TypeScript to Python Conversion Guide

A systematic approach for converting TypeScript/Node.js codebases into clean, idiomatic Python client libraries.

## Overview

This guide documents the methodology used to convert MCP servers (TypeScript) into pure Python libraries with consistent design, quality, and architecture.

---

## Phase 1: Analysis

### 1.1 Understand the Source

Before writing any code, thoroughly analyze the TypeScript source:

```bash
# Find all TypeScript files
find ./source -name "*.ts" -type f

# Count lines to estimate complexity
wc -l ./source/**/*.ts

# Identify entry points
grep -r "export class\|export function\|export default" ./source
```

**Key questions:**
- What is the core functionality? (API client, data processor, etc.)
- What external services does it connect to?
- What are the main public interfaces?
- What dependencies can be dropped (MCP, CLI frameworks)?

### 1.2 Map the Type System

Create a mapping document of TypeScript types to Python equivalents:

| TypeScript | Python |
|------------|--------|
| `interface Foo { ... }` | `@dataclass class Foo: ...` |
| `type Foo = { ... }` | `@dataclass class Foo: ...` |
| `enum Foo { A, B }` | `class Foo(str, Enum): ...` |
| `string \| null` | `str \| None` |
| `Promise<T>` | `Coroutine[Any, Any, T]` or just async return |
| `Record<string, T>` | `dict[str, T]` |
| `T[]` | `list[T]` |
| `Partial<T>` | All fields optional with `= None` |
| `readonly` | `frozen=True` on dataclass |

### 1.3 Identify Dependencies to Replace

| TypeScript Dependency | Python Replacement |
|-----------------------|-------------------|
| `axios` | `httpx` |
| `node-fetch` | `httpx` |
| `commander` / `yargs` | Remove (library, not CLI) |
| `better-sqlite3` | `sqlite3` (stdlib) |
| `@modelcontextprotocol/sdk` | Remove entirely |
| `dotenv` | `os.environ` |
| `winston` / `pino` | `logging` (stdlib) |

---

## Phase 2: Architecture Design

### 2.1 Standard Module Structure

Every converted library follows this structure:

```
libname/
├── __init__.py      # Public exports, docstring with usage examples
├── client.py        # Main client class + convenience functions
├── types.py         # All dataclasses and enums
├── result.py        # Result[T, E] type (Ok/Err)
├── config.py        # Optional: constants, defaults
├── pyproject.toml   # Package metadata
└── py.typed         # PEP 561 marker (empty file)
```

### 2.2 The Result Type Pattern

Always use Result types instead of exceptions for expected errors:

```python
# result.py
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool: return True
    def is_err(self) -> bool: return False
    def unwrap(self) -> T: return self.value
    def unwrap_or(self, default: T) -> T: return self.value

@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool: return False
    def is_err(self) -> bool: return True
    def unwrap(self) -> T: raise ValueError(f"Called unwrap on Err: {self.error}")
    def unwrap_or(self, default: T) -> T: return default

Result = Union[Ok[T], Err[E]]
```

### 2.3 Dataclass Conventions

```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)  # Always use both
class ResponseType:
    """Docstring explaining the type."""

    # Required fields first
    id: str
    name: str

    # Optional fields with defaults
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    # Never use mutable defaults directly
    # BAD:  items: list[str] = []
    # GOOD: items: list[str] = field(default_factory=list)
```

---

## Phase 3: Conversion Patterns

### 3.1 Converting TypeScript Classes

**TypeScript:**
```typescript
export class ApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(config: ClientConfig) {
    this.baseUrl = config.baseUrl || DEFAULT_URL;
    this.apiKey = config.apiKey || process.env.API_KEY;
  }

  async search(query: string): Promise<SearchResponse> {
    const response = await axios.get(`${this.baseUrl}/search`, {
      params: { q: query },
      headers: { Authorization: `Bearer ${this.apiKey}` }
    });
    return response.data;
  }
}
```

**Python:**
```python
@dataclass
class ApiClient:
    """API client for service X."""

    api_key: str | None = None
    base_url: str = DEFAULT_URL
    timeout: float = 30.0
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._resolved_key = self.api_key or os.environ.get("API_KEY")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._resolved_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: str) -> Result[SearchResponse, ApiError]:
        if not self._resolved_key:
            return Err(ApiError("API key not configured"))

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return Ok(self._parse_response(resp.json()))
        except httpx.TimeoutException:
            return Err(ApiError("Request timed out"))
        except httpx.HTTPStatusError as e:
            return Err(ApiError(f"HTTP {e.response.status_code}"))
```

### 3.2 Converting TypeScript Interfaces

**TypeScript:**
```typescript
interface SearchResult {
  id: string;
  title: string;
  description?: string;
  score: number;
  tags: string[];
  metadata: Record<string, unknown>;
}
```

**Python:**
```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    title: str
    score: float
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.3 Converting Enums

**TypeScript:**
```typescript
enum SearchType {
  AUTO = "auto",
  FAST = "fast",
  DEEP = "deep"
}
```

**Python:**
```python
class SearchType(str, Enum):
    """Search type options."""
    AUTO = "auto"
    FAST = "fast"
    DEEP = "deep"
```

### 3.4 Converting Promise Chains / Async

**TypeScript:**
```typescript
async function fetchWithRetry(url: string, retries: number = 3): Promise<Response> {
  for (let i = 0; i <= retries; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
    } catch (error) {
      if (i === retries) throw error;
      await new Promise(r => setTimeout(r, Math.exp(i) * 50));
    }
  }
  throw new Error("Exhausted retries");
}
```

**Python:**
```python
async def fetch_with_retry(
    url: str,
    retries: int = 3
) -> Result[httpx.Response, str]:
    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return Ok(response)
        except httpx.HTTPError as e:
            last_error = str(e)
            if attempt < retries:
                await asyncio.sleep(math.exp(attempt) * 0.05)

    return Err(last_error or "Exhausted retries")
```

### 3.5 Converting Error Handling

**TypeScript (exceptions):**
```typescript
if (!apiKey) {
  throw new Error("API key required");
}
```

**Python (Result type):**
```python
if not api_key:
    return Err(ConfigError("API key required"))
```

---

## Phase 4: HTTP Client Patterns

### 4.1 Basic httpx Usage

```python
import httpx

# Synchronous
def sync_request(url: str) -> Result[dict, str]:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return Ok(response.json())
    except httpx.HTTPError as e:
        return Err(str(e))

# Asynchronous
async def async_request(url: str) -> Result[dict, str]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return Ok(response.json())
    except httpx.HTTPError as e:
        return Err(str(e))
```

### 4.2 Request Configuration

```python
async def make_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Result[httpx.Response, ApiError]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
            )
            response.raise_for_status()
            return Ok(response)
    except httpx.TimeoutException:
        return Err(ApiError("Request timed out"))
    except httpx.HTTPStatusError as e:
        return Err(ApiError(f"HTTP {e.response.status_code}", e.response.status_code))
    except httpx.RequestError as e:
        return Err(ApiError(f"Request failed: {e}"))
```

---

## Phase 5: Package Configuration

### 5.1 pyproject.toml Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "libname"
version = "1.0.0"
description = "Python client library for Service X"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "author" }]
keywords = ["api", "client", "service-x"]
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
    "mypy>=1.0",
    "ruff>=0.1",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
```

### 5.2 __init__.py Template

```python
"""
libname - Python client library for Service X

Usage:
    from libname import Client

    client = Client(api_key="...")  # or set SERVICE_API_KEY env var

    result = await client.operation("arg")
    if result.is_ok():
        print(result.value)
    else:
        print(f"Error: {result.error}")
"""

from .client import Client, convenience_function
from .types import ResponseType, RequestType, ServiceError
from .result import Result, Ok, Err

__all__ = [
    "Client",
    "convenience_function",
    "ResponseType",
    "RequestType",
    "ServiceError",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"
```

---

## Phase 6: Quality Checklist

Before considering a conversion complete:

- [ ] All TypeScript types have Python dataclass equivalents
- [ ] All public functions return `Result[T, E]` not exceptions
- [ ] All dataclasses use `frozen=True, slots=True`
- [ ] No mutable default arguments (use `field(default_factory=...)`)
- [ ] Environment variable fallbacks for configuration
- [ ] Async methods use `httpx.AsyncClient`
- [ ] Timeouts on all HTTP requests
- [ ] Type hints on all public functions
- [ ] Docstrings with usage examples in `__init__.py`
- [ ] `py.typed` marker file exists
- [ ] `pyproject.toml` configured for Python 3.11+
- [ ] No MCP or CLI framework dependencies remain
- [ ] Imports are clean (no unused imports)

---

## Common Pitfalls

### 1. Don't Preserve TypeScript Idioms

```python
# BAD: TypeScript-style null checks
if api_key is not None and api_key != "":
    ...

# GOOD: Pythonic
if api_key:
    ...
```

### 2. Don't Use Exceptions for Control Flow

```python
# BAD
try:
    result = api_call()
except ApiError as e:
    return default_value

# GOOD
result = api_call()
if result.is_err():
    return default_value
```

### 3. Don't Forget Timeout Handling

```python
# BAD: No timeout
async with httpx.AsyncClient() as client:
    response = await client.get(url)

# GOOD: Explicit timeout
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
```

### 4. Don't Mix Sync and Async

```python
# BAD: Blocking call in async function
async def fetch():
    return requests.get(url)  # Blocks event loop!

# GOOD: Use async client
async def fetch():
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

# MCP to Skill Conversion

Methodology for wrapping Model Context Protocol (MCP) servers as Claude Skills.

## Why Convert MCP to Skills?

MCP servers provide tools via a standard protocol. Skills wrap those tools with:

| MCP Direct | Skill Wrapper |
|------------|---------------|
| LLM sees tool names | LLM sees skill commands |
| LLM chooses which server | Skill routes intelligently |
| Protocol details visible | Implementation hidden |
| One server = one capability | Multiple servers = one skill |

**When to wrap:**
- Combining multiple MCP servers into one coherent interface
- Adding routing logic (which backend for which query?)
- Implementing caching, fallbacks, or result merging
- Hiding credential management from the LLM

**When NOT to wrap:**
- Single MCP server with simple interface
- Direct tool access is sufficient
- Adding abstraction without clear benefit

## Conversion Process

### 1. Audit the MCP Server

Before converting, understand what you're wrapping:

```bash
# List available tools
npx @modelcontextprotocol/inspector <server-command>

# Or read the server's documentation
cat servers/<server>/README.md
```

Document:
- **Tools**: Names, parameters, return types
- **Transport**: stdio, HTTP, Docker?
- **Credentials**: Required environment variables
- **Limitations**: Rate limits, supported inputs

### 2. Design the Skill Interface

Map MCP tools to skill commands:

```
MCP Server: context7
├── resolve-library-id(libraryName) → library_id
└── get-library-docs(id, topic, mode) → docs

Skill: encyclopedia
├── search <query> → calls resolve-library-id + get-library-docs
└── lookup <topic> --library <name> → targeted search
```

**Design principles:**
- Fewer commands than underlying tools
- Compose multiple tools into single operations
- Hide intermediate steps (library ID resolution)

### 3. Create the Skill Structure

```
skill-name/
├── SKILL.md                 # Public interface
├── scripts/
│   ├── main.py              # CLI entry point
│   └── mcp_client.py        # MCP connection handling
├── config/
│   └── servers.json         # MCP server configuration
└── cache/                   # Response caching
```

### 4. Implement the MCP Client

Generic MCP client wrapper:

```python
"""mcp_client.py - Connects to MCP servers via stdio."""

import json
import subprocess
from dataclasses import dataclass
from typing import Any

@dataclass
class MCPResponse:
    success: bool
    data: Any
    error: str | None = None

class MCPClient:
    def __init__(self, command: list[str], env: dict = None):
        self.command = command
        self.env = env or {}
        self.process = None

    def connect(self) -> None:
        import os
        env = {**os.environ, **self.env}
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

    def call_tool(self, name: str, arguments: dict) -> MCPResponse:
        if not self.process:
            self.connect()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }

        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()

        line = self.process.stdout.readline()
        response = json.loads(line)

        if "error" in response:
            return MCPResponse(False, None, response["error"]["message"])

        return MCPResponse(True, response.get("result"))

    def close(self) -> None:
        if self.process:
            self.process.terminate()
            self.process = None
```

### 5. Implement Command Routing

Route skill commands to MCP tools:

```python
"""main.py - Skill CLI with MCP backend routing."""

import json
import sys
from pathlib import Path
from mcp_client import MCPClient, MCPResponse

SKILL_ROOT = Path(__file__).parent.parent
CONFIG = json.loads((SKILL_ROOT / "config/servers.json").read_text())

def get_client(server_name: str) -> MCPClient:
    """Get MCP client for named server."""
    server = CONFIG["servers"].get(server_name)
    if not server:
        raise ValueError(f"Unknown server: {server_name}")

    return MCPClient(
        command=server["command"],
        env=server.get("env", {})
    )

def search(query: str) -> dict:
    """Search command - routes to appropriate MCP server."""

    # Classify query to choose backend
    if is_library_query(query):
        return search_library(query)
    else:
        return search_web(query)

def search_library(query: str) -> dict:
    """Search library documentation via context7."""
    client = get_client("context7")

    # First resolve the library ID
    response = client.call_tool("resolve-library-id", {"libraryName": query})
    if not response.success:
        return {"success": False, "error": response.error}

    library_id = response.data.get("library_id")
    if not library_id:
        return {"success": False, "error": "Library not found"}

    # Then fetch docs
    response = client.call_tool("get-library-docs", {
        "context7CompatibleLibraryID": library_id,
        "topic": query,
        "mode": "code"
    })

    client.close()
    return sanitize_response(response)

def search_web(query: str) -> dict:
    """Search web via exa."""
    client = get_client("exa")

    response = client.call_tool("web_search_exa", {"query": query})

    client.close()
    return sanitize_response(response)

def sanitize_response(response: MCPResponse) -> dict:
    """Remove internal details from response."""
    if not response.success:
        return {"success": False, "error": response.error}

    # Strip server names, model IDs, etc.
    data = response.data
    # ... sanitization logic ...

    return {"success": True, "data": data}

def is_library_query(query: str) -> bool:
    """Classify query as library-specific or general."""
    library_keywords = ["docs", "api", "reference", "function", "class", "method"]
    return any(kw in query.lower() for kw in library_keywords)
```

### 6. Configure Servers

```json
{
    "servers": {
        "context7": {
            "command": ["npx", "-y", "@context7/mcp"],
            "env": {"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"},
            "capabilities": ["library_docs"],
            "optional": true
        },
        "exa": {
            "command": ["npx", "-y", "@exa/mcp"],
            "env": {"EXA_API_KEY": "${EXA_API_KEY}"},
            "capabilities": ["web_search", "code_search"],
            "optional": false
        }
    },
    "routing": {
        "library_docs": ["context7"],
        "web_search": ["exa"],
        "code_search": ["exa", "context7"]
    }
}
```

## Handling Multiple Backends

### Source Routing

Route queries to appropriate backends:

```python
def route_query(query: str, query_type: str) -> list[str]:
    """Determine which backends to query."""
    routing = CONFIG.get("routing", {})
    backends = routing.get(query_type, [])

    # Filter to available backends
    available = []
    for backend in backends:
        server = CONFIG["servers"].get(backend)
        if server and (not server.get("optional") or has_credentials(backend)):
            available.append(backend)

    return available
```

### Result Merging

Combine results from multiple sources:

```python
def merge_results(results: list[dict]) -> dict:
    """Merge and deduplicate results from multiple backends."""
    seen = set()
    merged = []

    for result in results:
        if not result.get("success"):
            continue

        for item in result.get("data", []):
            # Dedupe by content hash
            key = hash_content(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)

    return {"success": True, "data": merged, "count": len(merged)}

def hash_content(item: dict) -> str:
    """Generate hash for deduplication."""
    import hashlib
    content = json.dumps(item, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Fallback Chains

Handle backend failures gracefully:

```python
async def search_with_fallback(query: str) -> dict:
    """Try backends in priority order until one succeeds."""
    backends = route_query(query, "web_search")
    errors = []

    for backend in backends:
        try:
            client = get_client(backend)
            result = await client.call_tool("search", {"query": query})
            client.close()

            if result.success:
                return sanitize_response(result)
            else:
                errors.append(f"{backend}: {result.error}")

        except Exception as e:
            errors.append(f"{backend}: {e}")
            continue

    return {
        "success": False,
        "error": f"All backends failed: {'; '.join(errors)}"
    }
```

## Credential Management

### Loading Credentials

```python
import os
from pathlib import Path

def load_credentials() -> dict:
    """Load credentials from .env.local or environment."""
    creds = {}

    # Try .env.local first
    env_file = Path.home() / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                creds[key.strip()] = value.strip()

    # Environment variables override
    for key in ["CONTEXT7_API_KEY", "EXA_API_KEY", "PERPLEXITY_API_KEY"]:
        if key in os.environ:
            creds[key] = os.environ[key]

    return creds
```

### Credential Validation

```python
import re

CREDENTIAL_PATTERNS = {
    "OPENAI_API_KEY": r"^sk-(proj-)?[a-zA-Z0-9-_]+$",
    "ANTHROPIC_API_KEY": r"^sk-ant-[a-zA-Z0-9-_]+$",
    "EXA_API_KEY": r"^[a-f0-9-]{36}$",
}

def validate_credentials(creds: dict) -> list[str]:
    """Validate credential formats, return list of errors."""
    errors = []

    for key, pattern in CREDENTIAL_PATTERNS.items():
        if key in creds:
            if not re.match(pattern, creds[key]):
                errors.append(f"{key} has invalid format")

    return errors
```

### Credential Masking

```python
def mask_credential(value: str) -> str:
    """Mask credential for logging."""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"

def sanitize_for_logging(data: dict) -> dict:
    """Remove credentials from data before logging."""
    import copy

    result = copy.deepcopy(data)
    sensitive_keys = ["key", "token", "secret", "password", "credential"]

    def redact(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(s in k.lower() for s in sensitive_keys):
                    obj[k] = "***REDACTED***"
                else:
                    redact(v)
        elif isinstance(obj, list):
            for item in obj:
                redact(item)

    redact(result)
    return result
```

## Caching

### Response Caching

```python
import time
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_TTL = 3600  # 1 hour

def get_cache_key(tool: str, args: dict) -> str:
    """Generate cache key from tool call."""
    import hashlib
    content = json.dumps({"tool": tool, "args": args}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def get_cached(tool: str, args: dict) -> dict | None:
    """Get cached response if valid."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = get_cache_key(tool, args)
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    data = json.loads(cache_file.read_text())
    if time.time() - data.get("timestamp", 0) > CACHE_TTL:
        cache_file.unlink()  # Expired
        return None

    return data.get("response")

def set_cached(tool: str, args: dict, response: dict) -> None:
    """Cache a response."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = get_cache_key(tool, args)
    cache_file = CACHE_DIR / f"{key}.json"

    cache_file.write_text(json.dumps({
        "timestamp": time.time(),
        "response": response
    }))
```

## Connection Pooling

For skills calling multiple MCP servers:

```python
from threading import Lock

class MCPPool:
    """Connection pool for MCP clients."""

    def __init__(self, max_per_server: int = 3):
        self.max_per_server = max_per_server
        self.pools: dict[str, list[MCPClient]] = {}
        self.locks: dict[str, Lock] = {}

    def get_client(self, server_name: str) -> MCPClient:
        """Get or create a client for the server."""
        if server_name not in self.pools:
            self.pools[server_name] = []
            self.locks[server_name] = Lock()

        with self.locks[server_name]:
            # Return idle client if available
            if self.pools[server_name]:
                return self.pools[server_name].pop()

            # Create new if under limit
            if len(self.pools[server_name]) < self.max_per_server:
                server = CONFIG["servers"][server_name]
                client = MCPClient(server["command"], server.get("env"))
                client.connect()
                return client

        # Wait and retry if at limit
        raise RuntimeError(f"Connection pool exhausted for {server_name}")

    def return_client(self, server_name: str, client: MCPClient) -> None:
        """Return client to pool."""
        with self.locks[server_name]:
            self.pools[server_name].append(client)
```

## Testing Converted Skills

### Unit Tests

```python
def test_routing():
    """Test query routing logic."""
    assert is_library_query("React hooks documentation") == True
    assert is_library_query("weather in London") == False

def test_sanitization():
    """Test response sanitization."""
    raw = {"server": "context7", "model": "gpt-4", "result": "data"}
    clean = sanitize_response(MCPResponse(True, raw))
    assert "server" not in clean
    assert "model" not in clean

def test_credential_masking():
    """Test credential masking."""
    assert mask_credential("sk-1234567890abcdef") == "sk-1...cdef"
    assert mask_credential("short") == "***"
```

### Integration Tests

```python
def test_search_library():
    """Test library search end-to-end."""
    result = search("React useState hook")
    assert result["success"] == True
    assert "data" in result

def test_fallback_chain():
    """Test fallback when primary backend fails."""
    # Mock primary to fail
    with mock_backend_failure("context7"):
        result = search("Python requests library")
        assert result["success"] == True  # Should fall back to exa
```

## Common Pitfalls

### 1. Blocking on MCP Connections

MCP servers may hang. Always use timeouts:

```python
import asyncio

async def call_with_timeout(client, tool, args, timeout=30):
    try:
        return await asyncio.wait_for(
            client.call_tool(tool, args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return MCPResponse(False, None, f"Timeout after {timeout}s")
```

### 2. Credential Leakage

Check all output paths:

```python
def respond(success, data, message):
    # Sanitize before output
    safe_data = sanitize_for_logging(data) if data else None
    safe_message = sanitize_for_logging({"m": message})["m"]

    print(json.dumps({
        "success": success,
        "data": safe_data,
        "message": safe_message
    }))
```

### 3. Resource Exhaustion

Limit concurrent connections:

```python
MAX_CONCURRENT = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def limited_call(client, tool, args):
    async with semaphore:
        return await client.call_tool(tool, args)
```

---

*For architectural patterns when combining multiple skills, see [Cognitive Construct Design](cognitive-construct-design.md).*

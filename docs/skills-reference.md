# Skills Reference

Comprehensive reference for Claude Skills—their structure, behavior, and integration patterns.

## Core Concepts

### What Skills Solve

LLMs face a fundamental tradeoff: **capability vs. context**. Loading all possible knowledge into the system prompt wastes tokens and dilutes focus. Skills solve this by:

1. **Lazy loading** — Instructions load only when needed
2. **Progressive disclosure** — Details emerge as tasks require them
3. **Domain encapsulation** — Specialized knowledge stays packaged and portable

### The Progressive Disclosure Pattern

Skills reveal themselves in layers:

```
Layer 0: Name + Short Description (always visible)
    ↓ (skill becomes relevant)
Layer 1: Full Instructions (loaded on invocation)
    ↓ (specific resource needed)
Layer 2: Resources, scripts, data (loaded on demand)
```

This mirrors how human expertise works: you know someone "does accounting" before you ask them to explain GAAP depreciation rules.

### Opacity and Abstraction

Skills present **opaque interfaces**—the LLM interacts with high-level commands without seeing implementation details:

```
User request: "Find documentation for React hooks"
    ↓
Skill interface: search("React hooks")
    ↓
Hidden: Route to context7 → parse response → cache result
    ↓
LLM sees: "Found 3 relevant docs on useState, useEffect, useContext"
```

The LLM doesn't know (or need to know) which backend answered. This enables:
- **Backend swapping** without changing the interface
- **Graceful degradation** when services fail
- **Reduced prompt complexity** for the LLM

## Skill Anatomy

### Required: SKILL.md

Every skill needs a `SKILL.md` file with YAML frontmatter:

```yaml
---
name: my-skill              # kebab-case, max 64 chars
description: Brief purpose  # max 200 chars, shown in discovery
version: 1.0.0
dependencies: python>=3.8   # optional
license: MIT                # optional
---

# My Skill

Detailed instructions go here. This section loads when the skill
is invoked, not during discovery.

## Commands

- `command1 <arg>` — What it does
- `command2 <arg>` — What it does

## Usage Notes

Any context the LLM needs to use this skill effectively.
```

**Why the constraints?**
- `name` ≤64 chars: Fits in tool registries and logs
- `description` ≤200 chars: Prevents context bloat during discovery
- Frontmatter before prose: Parseable metadata, human-readable docs

### Directory Structure

```
skill-name/
├── SKILL.md              # Required: metadata + instructions
├── scripts/              # Executable tools (optional)
│   ├── main.py           # Entry point
│   └── helpers.py        # Supporting code
├── resources/            # Reference materials (optional)
│   ├── config.json       # Configuration
│   └── templates/        # Output templates
├── cache/                # Runtime cache (optional)
└── sessions/             # Persistent state (optional)
```

### Scripts

Skills can include executable scripts that Claude invokes:

```python
#!/usr/bin/env python3
"""skill-name/scripts/main.py"""

import json
import sys
from dataclasses import dataclass, asdict

@dataclass
class SkillResponse:
    success: bool
    data: any
    message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

def main():
    if len(sys.argv) < 2:
        print(SkillResponse(False, None, "Usage: main.py <command> [args]").to_json())
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Route to command handlers
    handlers = {
        "search": handle_search,
        "lookup": handle_lookup,
    }

    handler = handlers.get(command)
    if not handler:
        print(SkillResponse(False, None, f"Unknown command: {command}").to_json())
        sys.exit(1)

    result = handler(*args)
    print(result.to_json())

if __name__ == "__main__":
    main()
```

**Design principles:**
- JSON output (machine-parseable)
- Exit codes (0 = success, non-zero = failure)
- No interactive prompts
- Errors returned as values, not exceptions

## API Integration

### Messages API with Skills

```python
from anthropic import Anthropic

client = Anthropic()

# Using built-in skills
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Search the web for Python 3.13 features"}],
    skills=[
        {"type": "web_search_2025_01_24"}
    ]
)

# Skills with configuration
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Analyze this spreadsheet"}],
    skills=[
        {
            "type": "computer_use_2025_01_24",
            "display_width": 1920,
            "display_height": 1080
        }
    ]
)
```

### Streaming with Skills

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Process this document"}],
    skills=[{"type": "text_editor_2025_01_24"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Tool Results and Skill Interactions

When skills invoke tools, handle results in the conversation:

```python
messages = [{"role": "user", "content": "Find and summarize recent AI papers"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=messages,
        skills=[{"type": "web_search_2025_01_24"}]
    )

    if response.stop_reason == "end_turn":
        break

    if response.stop_reason == "tool_use":
        # Process tool calls from skills
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

## Built-in Skills

### computer_use

GUI automation via screenshots and synthetic input.

```python
skills=[{
    "type": "computer_use_2025_01_24",
    "display_width": 1920,
    "display_height": 1080
}]
```

**Capabilities:** Screenshot capture, mouse/keyboard events, window management

### text_editor

File manipulation with structured operations.

```python
skills=[{"type": "text_editor_2025_01_24"}]
```

**Capabilities:** Read, write, insert, replace, undo/redo

### web_search

Real-time web queries with source attribution.

```python
skills=[{"type": "web_search_2025_01_24"}]
```

**Capabilities:** Search queries, URL fetching, content extraction

### code_execution

Sandboxed code running with output capture.

```python
skills=[{"type": "code_execution_2025_01_24"}]
```

**Capabilities:** Python execution, package installation, file I/O within sandbox

## Response Handling

### Standard Response Format

Skills should return consistent JSON:

```json
{
    "success": true,
    "data": {
        "results": [...],
        "metadata": {...}
    },
    "message": "Found 5 matching documents"
}
```

### Error Response Format

```json
{
    "success": false,
    "error": {
        "code": "BACKEND_UNAVAILABLE",
        "message": "Search service temporarily unavailable",
        "recoverable": true
    },
    "metadata": {
        "skill": "encyclopedia",
        "command": "search",
        "timestamp": "2024-12-14T10:30:00Z"
    }
}
```

**Error categories:**
- `CONFIG_ERROR` — Missing credentials, invalid settings (exit 1)
- `BACKEND_ERROR` — Service unavailable, API failure (exit 2)
- `INPUT_ERROR` — Invalid command, malformed query (exit 3)
- `INTERNAL_ERROR` — Unexpected failure (exit 4)

## Security Considerations

### Credential Management

- Store credentials in `.env.local` or environment variables
- Never expose credentials in skill output
- Validate credential format before use
- Mask credentials in logs: `sk-...abc123` → `sk-***`

```python
import re

CREDENTIAL_PATTERNS = {
    "OPENAI_API_KEY": r"^sk-(proj-)?[a-zA-Z0-9-_]+$",
    "ANTHROPIC_API_KEY": r"^sk-ant-[a-zA-Z0-9-_]+$",
}

def validate_credential(name: str, value: str) -> bool:
    pattern = CREDENTIAL_PATTERNS.get(name)
    if not pattern:
        return True  # No validation for unknown keys
    return bool(re.match(pattern, value))

def mask_credential(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"
```

### Input Validation

- Sanitize all user input before passing to backends
- Enforce size limits on queries and responses
- Rate limit operations to prevent abuse

### Output Sanitization

- Strip internal details (server names, model IDs, file paths)
- Remove stack traces from error messages
- Ensure no credential patterns in output

## Performance Patterns

### Caching

Cache expensive operations locally:

```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("cache")

def cached(fn):
    def wrapper(*args, **kwargs):
        key = hashlib.sha256(
            json.dumps({"args": args, "kwargs": kwargs}).encode()
        ).hexdigest()[:16]

        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        result = fn(*args, **kwargs)
        cache_file.write_text(json.dumps(result))
        return result
    return wrapper
```

### Connection Pooling

For skills managing multiple backends:

```python
CONNECTION_POOL = {
    "max_per_server": 3,
    "timeout_ms": 5000,
    "idle_timeout_ms": 60000,
}
```

### Graceful Degradation

When backends fail, fall back gracefully:

```python
async def search(query: str) -> SkillResponse:
    backends = ["primary", "secondary", "fallback"]

    for backend in backends:
        try:
            result = await call_backend(backend, query)
            return SkillResponse(True, result, f"Found via {backend}")
        except BackendError:
            continue

    return SkillResponse(False, None, "All backends unavailable")
```

---

*See [Creating Skills](creating-skills.md) for a step-by-step guide to building custom skills.*

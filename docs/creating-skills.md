# Creating Custom Skills

Step-by-step guide to building Skills that extend Claude with domain-specific capabilities.

## Planning Your Skill

Before writing code, answer these questions:

### 1. What problem does this skill solve?

A good skill should:
- **Encapsulate domain knowledge** — Coding standards, legal requirements, company processes
- **Wrap complex tools** — APIs, databases, external services
- **Standardize workflows** — Multi-step procedures that benefit from consistency

A skill probably shouldn't:
- Duplicate Claude's existing capabilities
- Wrap trivial operations (just use the tool directly)
- Require real-time human interaction

### 2. What's the interface?

Design commands that are:
- **Verb-first**: `search <query>`, `analyze <file>`, `generate <type>`
- **Self-documenting**: Command names should imply their function
- **Composable**: Small commands that chain together

### 3. What state does it need?

| State Type | Storage | Example |
|------------|---------|---------|
| None | — | Pure transformation skills |
| Session | Memory (lost on restart) | Conversation context |
| Persistent | Files (JSON, SQLite) | User preferences, learned patterns |
| External | APIs, databases | Shared knowledge bases |

## Minimal Skill Structure

Start with the simplest possible skill:

```
my-skill/
├── SKILL.md
└── scripts/
    └── main.py
```

### SKILL.md Template

```markdown
---
name: my-skill
description: One-line description of what this skill does
version: 1.0.0
---

# My Skill

Brief explanation of the skill's purpose and when to use it.

## Commands

- `command1 <arg>` — Description of what it does
- `command2 <arg> [--option]` — Description with optional flag

## Examples

```bash
# Basic usage
python3 scripts/main.py command1 "input"

# With options
python3 scripts/main.py command2 "input" --option value
```

## Notes

Any caveats, limitations, or special considerations.
```

### main.py Template

```python
#!/usr/bin/env python3
"""Minimal skill entry point."""

import json
import sys
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Response:
    success: bool
    data: Any
    message: str

def respond(success: bool, data: Any, message: str) -> None:
    print(json.dumps(asdict(Response(success, data, message)), indent=2))
    sys.exit(0 if success else 1)

def command1(arg: str) -> None:
    # Your implementation here
    result = {"processed": arg}
    respond(True, result, f"Processed: {arg}")

def command2(arg: str, option: str = None) -> None:
    # Your implementation here
    result = {"arg": arg, "option": option}
    respond(True, result, "Command2 executed")

def main() -> None:
    if len(sys.argv) < 2:
        respond(False, None, "Usage: main.py <command> [args]")

    command = sys.argv[1]
    args = sys.argv[2:]

    handlers = {
        "command1": lambda: command1(args[0]) if args else respond(False, None, "command1 requires an argument"),
        "command2": lambda: command2(args[0], args[1] if len(args) > 1 else None) if args else respond(False, None, "command2 requires an argument"),
    }

    handler = handlers.get(command)
    if not handler:
        respond(False, None, f"Unknown command: {command}. Available: {', '.join(handlers.keys())}")

    handler()

if __name__ == "__main__":
    main()
```

## Adding Complexity

### Configuration Files

For skills with settings:

```
my-skill/
├── SKILL.md
├── scripts/
│   └── main.py
└── config.json
```

```json
{
    "default_timeout": 30,
    "max_results": 10,
    "backends": ["primary", "fallback"]
}
```

```python
import json
from pathlib import Path

def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}  # Defaults
```

### Persistent State

For skills that remember things:

```python
from pathlib import Path
import json

STATE_FILE = Path(__file__).parent.parent / "state.json"

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"history": [], "preferences": {}}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

def remember(key: str, value: str) -> None:
    state = load_state()
    state["history"].append({"key": key, "value": value})
    save_state(state)
    respond(True, {"key": key}, f"Remembered: {key}")
```

### External API Integration

For skills wrapping external services:

```python
import os
import urllib.request
import json

def call_api(endpoint: str, params: dict) -> dict:
    api_key = os.environ.get("MY_API_KEY")
    if not api_key:
        respond(False, None, "MY_API_KEY not set in environment")

    url = f"https://api.example.com/{endpoint}"
    data = json.dumps(params).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        respond(False, None, f"API error: {e.code}")
    except urllib.error.URLError as e:
        respond(False, None, f"Network error: {e.reason}")
```

## Error Handling

### Error Categories

Use consistent error codes:

```python
class SkillError(Exception):
    def __init__(self, code: str, message: str, recoverable: bool = True):
        self.code = code
        self.message = message
        self.recoverable = recoverable

# Usage
raise SkillError("CONFIG_MISSING", "API key not found", recoverable=True)
raise SkillError("BACKEND_DOWN", "Service unavailable", recoverable=True)
raise SkillError("INVALID_INPUT", "Query cannot be empty", recoverable=True)
raise SkillError("INTERNAL", "Unexpected error", recoverable=False)
```

### Error Response Format

```python
def error_response(code: str, message: str, recoverable: bool = True) -> None:
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "recoverable": recoverable
        }
    }
    print(json.dumps(response, indent=2))
    sys.exit(1)
```

### Graceful Degradation

```python
def search_with_fallback(query: str) -> dict:
    backends = [
        ("primary", search_primary),
        ("secondary", search_secondary),
        ("cache", search_cache),
    ]

    errors = []
    for name, fn in backends:
        try:
            return fn(query)
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue

    error_response(
        "ALL_BACKENDS_FAILED",
        f"All search backends failed: {'; '.join(errors)}"
    )
```

## Testing Your Skill

### Manual Testing

```bash
# Test each command
python3 scripts/main.py command1 "test input"
python3 scripts/main.py command2 "test" --option value

# Test error cases
python3 scripts/main.py unknown_command
python3 scripts/main.py command1  # Missing argument

# Test with invalid environment
unset MY_API_KEY && python3 scripts/main.py command1 "test"
```

### Automated Testing

```python
# tests/test_skill.py
import subprocess
import json

def run_skill(command: str, *args) -> dict:
    result = subprocess.run(
        ["python3", "scripts/main.py", command, *args],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout), result.returncode

def test_command1_success():
    response, code = run_skill("command1", "test")
    assert code == 0
    assert response["success"] == True

def test_command1_missing_arg():
    response, code = run_skill("command1")
    assert code == 1
    assert response["success"] == False

def test_unknown_command():
    response, code = run_skill("nonexistent")
    assert code == 1
    assert "Unknown command" in response["message"]
```

## Best Practices

### Do

- **Return JSON** — Machine-parseable output
- **Use exit codes** — 0 for success, non-zero for failure
- **Fail fast** — Validate inputs before expensive operations
- **Log to stderr** — Keep stdout clean for responses
- **Document everything** — SKILL.md is your contract

### Don't

- **Don't use interactive prompts** — Skills run non-interactively
- **Don't expose credentials** — Mask in logs, never in output
- **Don't swallow errors** — Surface failures clearly
- **Don't assume state** — Skills may run in fresh environments
- **Don't block forever** — Use timeouts on all I/O

### Security Checklist

- [ ] Credentials loaded from environment, not hardcoded
- [ ] Input sanitized before external calls
- [ ] Output stripped of internal paths and server names
- [ ] Error messages don't leak stack traces
- [ ] Rate limits enforced on expensive operations

## Example: Documentation Search Skill

Complete example of a skill that searches documentation:

```
doc-search/
├── SKILL.md
├── scripts/
│   └── search.py
├── config.json
└── cache/
```

**SKILL.md:**
```markdown
---
name: doc-search
description: Search documentation across multiple sources
version: 1.0.0
---

# Documentation Search

Unified documentation search across configured sources.

## Commands

- `search <query>` — Search all sources for query
- `sources` — List available documentation sources
- `clear-cache` — Clear the response cache

## Configuration

Edit `config.json` to add/remove documentation sources.
```

**scripts/search.py:**
```python
#!/usr/bin/env python3
import json
import sys
import hashlib
from pathlib import Path

SKILL_ROOT = Path(__file__).parent.parent
CONFIG_FILE = SKILL_ROOT / "config.json"
CACHE_DIR = SKILL_ROOT / "cache"

def respond(success, data, message):
    print(json.dumps({"success": success, "data": data, "message": message}, indent=2))
    sys.exit(0 if success else 1)

def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"sources": []}

def get_cache_key(query):
    return hashlib.sha256(query.encode()).hexdigest()[:16]

def search_cached(query):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None

def cache_result(query, result):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{get_cache_key(query)}.json"
    cache_file.write_text(json.dumps(result))

def search(query):
    # Check cache first
    cached = search_cached(query)
    if cached:
        respond(True, cached, f"Found {len(cached)} results (cached)")

    config = load_config()
    results = []

    for source in config.get("sources", []):
        # In reality, this would call the source's API
        results.append({
            "source": source["name"],
            "title": f"Result for '{query}' from {source['name']}",
            "url": f"{source['base_url']}/search?q={query}"
        })

    cache_result(query, results)
    respond(True, results, f"Found {len(results)} results")

def sources():
    config = load_config()
    respond(True, config.get("sources", []), "Available sources")

def clear_cache():
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
    respond(True, None, "Cache cleared")

def main():
    if len(sys.argv) < 2:
        respond(False, None, "Usage: search.py <command> [args]")

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "search":
        if not args:
            respond(False, None, "search requires a query")
        search(" ".join(args))
    elif cmd == "sources":
        sources()
    elif cmd == "clear-cache":
        clear_cache()
    else:
        respond(False, None, f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
```

**config.json:**
```json
{
    "sources": [
        {"name": "python", "base_url": "https://docs.python.org/3"},
        {"name": "mdn", "base_url": "https://developer.mozilla.org"},
        {"name": "rust", "base_url": "https://doc.rust-lang.org"}
    ]
}
```

---

*For converting existing MCP servers to Skills, see [MCP to Skill Conversion](mcp-skill-conversion.md).*

# Volition

> Agency and execution. Edit code semantically, invoke LLMs, search the web, and query security services.

## Overview

**Volition** is the executive skill that transforms intent into action. It provides a unified interface for:

- **Semantic code editing** via LSP-powered tools (Serena)
- **LLM consultation** with automatic provider routing (cross-llm-mcp)
- **Web search** with reasoning models (openai-websearch-mcp)
- **Security reconnaissance** with safety constraints (mcp-shodan)

## Installation

```bash
# Install dependencies
pip install httpx pydantic
```

## Configuration

Set API keys in `.env.local` or environment:

```bash
# LLM Consultation (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...

# Security Queries (optional)
SHODAN_API_KEY=...
```

## Usage

### Check Available Capabilities

```bash
python3 volition.py capabilities
```

### Execute Actions (Auto-Routed)

```bash
python3 volition.py act "refactor the authentication module for better security"
python3 volition.py act "explain how this codebase handles errors"
python3 volition.py act "search for latest Python 3.13 features"
```

### Semantic Code Editing

```bash
python3 volition.py edit "UserAuth.validate" "add rate limiting check at start"
python3 volition.py edit "DatabasePool.connect" "add connection timeout" --project ./myapp
```

### Query Services

```bash
# Web search
python3 volition.py query web "latest Python 3.13 features"

# LLM consultation
python3 volition.py query llm "best practices for JWT token rotation" --tag coding

# Security query (requires --confirm)
python3 volition.py query security "exposed MongoDB instances" --confirm
```

## Output Format

All commands return structured JSON:

```json
{
  "status": "success",
  "handler": "web_search",
  "summary": "Found 5 results...",
  "data": { ... }
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid input |
| 2 | Resource not found |
| 3 | Backend unavailable |
| 4 | Permission denied |

## Backends

| Capability | Backend | Status |
|------------|---------|--------|
| Code Editing | [Serena](https://github.com/oraios/serena) | LSP-powered |
| LLM Consultation | cross-llm-mcp | Multi-provider |
| Web Search | openai-websearch-mcp | Reasoning models |
| Security Queries | mcp-shodan | Rate-limited |

## License

MIT

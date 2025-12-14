# OpenAI Web Search CLI

CLI-first tool for web search using OpenAI's reasoning models.

## Installation

```bash
# Using uv (recommended)
uv tool install openai-websearch-mcp

# Using pip
pip install openai-websearch-mcp
```

## Usage

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Basic search
openai-websearch search "latest AI news"

# Deep research with high reasoning effort
openai-websearch search "quantum computing advances" -m gpt-5 -e high

# Fast iteration with low effort
openai-websearch search "weather today" -m gpt-5-mini -e low

# Localized search
openai-websearch search "local events" --city "San Francisco" --country "US" --timezone "America/Los_Angeles"

# JSON output
openai-websearch search "tech news" -o json
```

## Commands

### search

Search the web using OpenAI's reasoning models.

```
openai-websearch search [OPTIONS] QUERY
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | AI model (gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, gpt-5-nano, o3, o4-mini) | `gpt-5-mini` |
| `-e, --effort` | Reasoning effort (low, medium, high, minimal) | Smart default |
| `--context` | Search context size (low, medium, high) | `medium` |
| `--city` | City for localized results | |
| `--country` | Country for localized results | |
| `--region` | Region for localized results | |
| `--timezone` | Timezone (required with --city) | |
| `-o, --output` | Output format (text, json) | `text` |

### install

Install in Claude Desktop configuration (macOS/Windows only).

```bash
openai-websearch install --api-key "sk-..." --default-model gpt-5-mini
```

## Model Selection

| Model | Reasoning | Default Effort | Best For |
|-------|-----------|----------------|----------|
| `gpt-4o` | ❌ | N/A | Standard search |
| `gpt-4o-mini` | ❌ | N/A | Basic queries |
| `gpt-5-mini` | ✅ | `low` | Fast iterations |
| `gpt-5` | ✅ | `medium` | Deep research |
| `gpt-5-nano` | ✅ | `medium` | Balanced |
| `o3` | ✅ | `medium` | Advanced reasoning |
| `o4-mini` | ✅ | `medium` | Efficient reasoning |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_DEFAULT_MODEL` | Default model | `gpt-5-mini` |

## Python API

```python
from openai_websearch_mcp import web_search, UserLocation

# Basic search
result = web_search("latest AI news")

# With options
result = web_search(
    "local tech events",
    model="gpt-5",
    reasoning_effort="high",
    user_location=UserLocation(
        city="San Francisco",
        country="US",
        timezone="America/Los_Angeles",
    ),
)
```

## License

MIT

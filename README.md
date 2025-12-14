# Cognitive Construct

Four cognitive skills that extend Claude with knowledge retrieval, persistent memory, structured reasoning, and agency.

## Skills

| Skill | Description |
|-------|-------------|
| [**Encyclopedia**](encyclopedia/) | Knowledge retrieval from multiple sources with intelligent routing |
| [**Inland Empire**](inland-empire/) | Unified memory substrate for persistent context across sessions |
| [**Rhetoric**](rhetoric/) | Structured reasoning with thought tracking and multi-model deliberation |
| [**Volition**](volition/) | Agency and execution with semantic code editing and LLM consultation |

## Quick Start

```bash
# Clone
git clone https://github.com/Raudbjorn/cognitive-construct.git
cd cognitive-construct

# Run setup wizard (configures API keys)
uv run scripts/setup.py

# Test API connectivity
uv run scripts/setup.py --test
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code                          │
│                  (Orchestrator)                         │
└───────────────────────┬─────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┬───────────────────┐
    │                   │                   │                   │
┌───▼───┐         ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
│ 📚    │         │    🧠     │       │    💭     │       │    ⚡     │
│ Ency- │         │  Inland   │       │ Rhetoric  │       │ Volition  │
│ clo-  │         │  Empire   │       │           │       │           │
│ pedia │         │           │       │           │       │           │
└───┬───┘         └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
    │                   │                   │                   │
Context7            LibSQL Graph      Sequential           Serena LSP
Exa                 Mem0 Cloud        Thinking             Cross-LLM
Perplexity          Session JSONL     AI Counsel           Web Search
Kagi                                  VibeCheck            Shodan
```

## Configuration

Each skill has its own `.env.local` file. The setup wizard auto-detects existing API keys and helps configure missing ones.

### Key APIs

| Skill | Required | Optional |
|-------|----------|----------|
| **Encyclopedia** | Exa or Perplexity | Context7, Kagi, SearXNG |
| **Inland Empire** | (none) | Mem0, LibSQL remote |
| **Rhetoric** | 2+ LLM keys | VibeCheck |
| **Volition** | 1+ LLM key | Shodan |

## Documentation

- [Skills Reference](docs/skills-reference.md) - Core concepts and usage
- [Creating Skills](docs/creating-skills.md) - Build custom skills
- [MCP Conversion](docs/mcp-skill-conversion.md) - Wrap MCP servers as skills
- [Design Doc](docs/cognitive-construct-design.md) - Architecture details

## Setup Commands

```bash
# Interactive setup
uv run scripts/setup.py

# Show configuration status
uv run scripts/setup.py --status

# Test API connectivity
uv run scripts/setup.py --test

# Auto-detect and show environment variables
uv run scripts/setup.py --detect

# Export as shell commands
uv run scripts/setup.py --export
```

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## License

MIT

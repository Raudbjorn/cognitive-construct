# Claude Skills Documentation

Reference documentation for Claude Skills—reusable capability packages that extend Claude with domain-specific knowledge and tools.

## Contents

| Document | Description |
|----------|-------------|
| [Skills Reference](skills-reference.md) | Core concepts, anatomy, and usage patterns |
| [Creating Skills](creating-skills.md) | Guide to building custom Skills |
| [MCP to Skill Conversion](mcp-skill-conversion.md) | Methodology for wrapping MCP servers as Skills |
| [Cognitive Construct Design](cognitive-construct-design.md) | Architecture for multi-skill AI enhancement systems |

## What Are Skills?

Skills are packages of instructions, resources, and tools that give Claude specialized capabilities. They follow a **progressive disclosure** pattern:

1. **Name + Description** — Loaded always (for capability discovery)
2. **Instructions** — Loaded when the skill is invoked
3. **Resources** — Loaded on demand as needed

Skills solve the context window problem: instead of stuffing all knowledge into the system prompt, Claude loads only what's needed for the current task.

## Skill Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Documents** | File format handling | PDF extraction, spreadsheet analysis |
| **Knowledge** | Domain expertise | Coding standards, legal research |
| **Tools** | External integrations | API wrappers, database queries |
| **Workflows** | Multi-step procedures | Code review, deployment pipelines |

## Quick Start

```python
# Using Skills via API
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Analyze this PDF"}],
    skills=[{"type": "computer_use_2025_01_24"}]  # Built-in skill
)
```

```bash
# Custom skill structure
my-skill/
├── SKILL.md          # Metadata + instructions (required)
├── scripts/          # Executable tools
└── resources/        # Reference materials
```

## Key Concepts

### Progressive Disclosure
Skills aren't dumped into context wholesale. Claude sees skill names first, loads instructions when relevant, and fetches resources only when needed. This keeps context focused.

### Opaque Interfaces
Skills abstract implementation details. The LLM interacts with high-level commands (`search`, `remember`, `deliberate`) without knowing which backend services handle the request.

### File-Based Persistence
Skills can maintain state across sessions through local files (JSON, SQLite, etc.), enabling memory, learning, and accumulated context.

## Built-in Skills

Anthropic provides several production-ready skills:

- **computer_use** — GUI automation and screenshot analysis
- **text_editor** — File manipulation with undo/redo
- **web_search** — Real-time web queries
- **code_execution** — Sandboxed code running

Custom skills extend this foundation with domain-specific capabilities.

---

*For implementation details, see the individual documents linked above.*

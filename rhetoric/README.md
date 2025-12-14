# Rhetoric: The Reasoning Engine

> "The unexamined thought is not worth having."

Rhetoric provides structured reasoning capabilities for complex problem-solving. It records thoughts with full revision tracking, orchestrates multi-model deliberation for decisions, and reviews thinking patterns to detect cognitive biases.

## Quick Start

```bash
# Check system status
python3 rhetoric/scripts/rhetoric.py status

# Record a thought
python3 rhetoric/scripts/rhetoric.py think "We should use a factory pattern here" --session-id project-123

# Deliberate on a question (requires 2+ model API keys)
python3 rhetoric/scripts/rhetoric.py deliberate "Should we use microservices or monolith?" --rounds 2

# Review thinking patterns
python3 rhetoric/scripts/rhetoric.py review --session-id project-123
```

## Commands

### `think`
Record a thought with structured metadata. Supports revision tracking and branching.

```bash
python3 rhetoric/scripts/rhetoric.py think "<thought>" [--session-id <id>] [--revision-of <id>] [--branch-from <id>]
```

### `deliberate`
Orchestrate multi-model debate on a question. Requires at least 2 configured API keys.

```bash
python3 rhetoric/scripts/rhetoric.py deliberate "<question>" [--rounds <1-5>] [--context <text>]
```

### `review`
Analyze thinking patterns and detect cognitive biases using VibeCheck integration.

```bash
python3 rhetoric/scripts/rhetoric.py review [--session-id <id>]
```

### `status`
Show system status including available models and deliberation readiness.

```bash
python3 rhetoric/scripts/rhetoric.py status
```

## Configuration

### Required Environment Variables

For deliberation, configure at least 2 of these API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=AIza...         # Google AI Studio key (also accepts GOOGLE_CLOUD_API_KEY)
OLLAMA_URL=http://localhost:11434
```

### Optional

```bash
USE_LEARNING_HISTORY=true  # Enable vibe-check learning history
```

## Directory Structure

```
rhetoric/
├── scripts/
│   └── rhetoric.py          # Main CLI entrypoint
├── ai-counsel/              # Multi-model deliberation engine
│   ├── ai_counsel/          # Client library
│   ├── adapters/            # HTTP adapters (OpenAI, Anthropic, etc.)
│   ├── deliberation/        # Deliberation engine
│   └── models/              # Pydantic schemas
├── sequentialthinking_py/   # Sequential thinking client
├── vibecheck_py/            # Pattern detection client
├── thoughts.json            # Thought persistence
├── deliberations/           # Deliberation transcripts
├── sessions/                # Session state
├── constitution.md          # Reasoning rules
└── learnings.json           # VibeCheck learnings
```

## Supported Adapters

| Adapter | Type | Models |
|---------|------|--------|
| **openai** | HTTP | gpt-4o, gpt-4o-mini, etc. |
| **anthropic** | HTTP | claude-sonnet-4-*, claude-3-*, etc. |
| **openrouter** | HTTP | Any model via OpenRouter |
| **ollama** | HTTP | Local models (llama3.2, etc.) |
| **lmstudio** | HTTP | Local LM Studio models |

## Features

- **Thought Recording**: Track thoughts with revision history and branching
- **Multi-Model Deliberation**: OpenAI + Anthropic + OpenRouter + Ollama debate
- **Convergence Detection**: Automatic consensus detection with similarity scoring
- **Pattern Analysis**: VibeCheck integration detects cognitive biases
- **Transcript Generation**: Markdown transcripts of deliberations

## Example Output

### Status
```json
{
  "active_sessions": 5,
  "total_thoughts": 42,
  "models_available": 4,
  "providers": ["openai", "anthropic", "openrouter", "gemini"],
  "deliberation_ready": true
}
```

### Deliberation
```json
{
  "status": "completed",
  "question": "Should we use microservices or monolith?",
  "rounds_completed": 2,
  "consensus": "Monolith recommended for MVP with clear module boundaries",
  "confidence": 0.85
}
```

### Review
```json
{
  "status": "success",
  "session_id": "project-123",
  "thought_count": 15,
  "patterns": {
    "revision_rate": 0.2,
    "branch_count": 2,
    "detected_issues": ["analysis_paralysis"]
  },
  "recommendations": ["Consider narrowing scope", "Make a decision to proceed"]
}
```

## Backends

Rhetoric orchestrates these components internally:

- **sequential-thinking**: Structured thought recording with chain-of-thought support
- **ai-counsel**: Multi-model deliberative consensus engine
- **vibe-check**: Pattern detection and cognitive bias assessment

## Dependencies

- Python >= 3.10
- httpx
- pydantic
- tenacity (for retry logic)
- PyYAML (for config)

## License

MIT

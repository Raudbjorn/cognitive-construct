# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Counsel is a **Python client library** for multi-model deliberative consensus. Unlike parallel opinion gathering, models see each other's responses and refine positions across multiple rounds of debate.

**Key differentiation**: Models engage in actual debate with cross-pollination (not just parallel aggregation).

**Production Ready**: 130+ passing tests, type-safe validation, graceful error handling, structured logging, convergence detection, AI-powered summaries, evidence-based deliberation with secure tool execution.

## Architecture

### Package Structure

```
ai_counsel/
├── __init__.py              # Public API, exports, __version__
├── client.py                # AICounselClient, convenience functions
├── result.py                # Ok, Err, Result types (error-as-values)
├── types.py                 # Frozen dataclasses (public API types)
├── py.typed                 # PEP 561 marker
└── _internal/
    ├── config.py            # Pydantic config validation
    └── adapters/            # HTTP adapters (ollama, lmstudio, openrouter)

# Legacy modules (being migrated)
adapters/                    # HTTP adapters used by engine
deliberation/                # Core deliberation engine
decision_graph/              # Optional memory system
models/                      # Pydantic schemas
```

### Core Design Principles

1. **Errors as values** - All public methods return `Result[T, E]` types
2. **Immutable by default** - Frozen dataclasses with slots for public types
3. **Async-first** - httpx for HTTP, asyncio for concurrency
4. **Type-safe** - Full type hints, Pydantic validation internally
5. **HTTP-only** - No CLI adapters, pure HTTP-based model access

### Public API

```python
from ai_counsel import AICounselClient, Participant, deliberate, Ok, Err

# Using client (recommended for multiple deliberations)
client = AICounselClient(
    openrouter_api_key="sk-...",
    ollama_url="http://localhost:11434",
)

result = await client.deliberate(
    question="Should we use TypeScript or JavaScript?",
    participants=[
        Participant(adapter="ollama", model="llama2"),
        Participant(adapter="openrouter", model="anthropic/claude-3.5-sonnet"),
    ],
    rounds=2,
)

if result.is_ok():
    print(result.value.summary.consensus)
else:
    print(f"Error: {result.error.message}")

# Using convenience function (for one-off deliberations)
result = await deliberate(question="...", participants=[...])
```

### HTTP Adapters

| Adapter | Type | Endpoint | Description |
|---------|------|----------|-------------|
| **Ollama** | HTTP | localhost:11434 | Local Ollama service |
| **LM Studio** | HTTP | localhost:1234 | Local LM Studio service |
| **OpenRouter** | HTTP | openrouter.ai | OpenRouter API gateway |

### Deliberation Engine (`deliberation/engine.py`)

- Orchestrates multi-round debates between models
- Manages context building from previous responses
- Coordinates convergence detection and early stopping
- Integrates tool execution system for evidence-based deliberation

### Convergence Detection (`deliberation/convergence.py`)

- Semantic similarity between consecutive rounds
- Backends: SentenceTransformer (best) → TF-IDF → Jaccard (zero deps)
- Statuses: converged (≥85%), refining (40-85%), diverging (<40%), impasse

### Decision Graph Memory (`decision_graph/`)

Optional persistent learning from deliberations with semantic search.

## Development Commands

### Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
pytest tests/unit -v                      # Unit tests
pytest tests/integration -v -m integration # Integration tests
pytest --cov=. --cov-report=html          # Coverage report
```

### Code Quality
```bash
black .           # Format
ruff check .      # Lint
mypy .            # Type check
```

## Configuration

### config.yaml

```yaml
version: "2.0"

adapters:
  ollama:
    type: http
    base_url: "http://localhost:11434"
    timeout: 300
    max_retries: 3

  openrouter:
    type: http
    base_url: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    timeout: 300
    max_retries: 3

defaults:
  mode: "quick"
  rounds: 2
  max_rounds: 5

decision_graph:
  enabled: true
  db_path: "decision_graph.db"
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_URL` | Ollama API URL |
| `LMSTUDIO_URL` | LM Studio API URL |
| `OPENROUTER_API_KEY` | OpenRouter API key |

## Type System

### Result Types (`result.py`)

```python
@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T
    def is_ok(self) -> bool: ...
    def unwrap(self) -> T: ...
    def map(self, fn: Callable[[T], U]) -> Result[U, E]: ...

@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E
    def is_err(self) -> bool: ...
    def unwrap_or(self, default: T) -> T: ...

Result = Union[Ok[T], Err[E]]
```

### Public Types (`types.py`)

All public types are frozen dataclasses:
- `Participant` - Adapter and model configuration
- `DeliberationRequest` - Request parameters
- `DeliberationResult` - Complete result
- `Vote`, `VotingResult` - Voting data
- `ConvergenceInfo` - Convergence metadata
- `SearchResult` - Decision graph query results
- `ApiError` - Error details

## Common Gotchas

1. **HTTP-only**: This library only supports HTTP adapters (no CLI tools)
2. **Result types**: All public methods return `Result[T, E]`, check `is_ok()` before accessing value
3. **Frozen dataclasses**: Public types are immutable
4. **Async-first**: All client methods are async
5. **Timeout Tuning**: Reasoning models can take 60-120+ seconds

## Testing Strategy

- **Unit Tests**: Mock adapters, test engine logic, convergence detection
- **Integration Tests**: Real adapter invocations (requires services running)
- **Fixtures**: `tests/conftest.py` provides shared fixtures

## Development Workflow

1. Write test first (TDD)
2. Implement feature
3. Run unit tests: `pytest tests/unit -v`
4. Format/lint: `black . && ruff check .`
5. Update CLAUDE.md if architecture changes

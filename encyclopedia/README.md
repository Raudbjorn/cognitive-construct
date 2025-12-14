# Encyclopedia Skill

Knowledge retrieval from multiple sources with intelligent routing.

## Overview

Encyclopedia aggregates multiple information sources into a unified interface:

- **Context7** - Library documentation lookup
- **Exa** - Web and code search
- **Perplexity** - AI-powered search
- **mcp-git-ingest** - Repository analysis
- **Kagi** - High-quality web search (optional)
- **SearXNG** - Meta search engine (optional)
- **CodeGraphContext** - Code graph analysis (optional)

The skill automatically routes queries to the most appropriate source, merges results, deduplicates using semantic similarity, and ranks by relevance.

## Commands

### search

Search across knowledge sources with intelligent routing.

```bash
python scripts/encyclopedia.py search "React hooks best practices"
python scripts/encyclopedia.py search "repo:anthropics/claude-code how does auth work"
python scripts/encyclopedia.py search "svelte runes" --sources context7,exa --limit 3
```

**Options:**
- `--sources` - Comma-separated list of sources to query
- `--limit` - Maximum results (default: 5)

### lookup

Look up documentation for a specific library or API.

```bash
python scripts/encyclopedia.py lookup "fastapi"
python scripts/encyclopedia.py lookup "react" --version 18
```

**Options:**
- `--version` - Specific version to look up

### code

Analyze a code repository.

```bash
python scripts/encyclopedia.py code "github.com/sveltejs/svelte" "what are runes"
python scripts/encyclopedia.py code "anthropics/claude-code" "authentication flow" --depth deep
```

**Options:**
- `--depth` - Analysis depth: `shallow` (default) or `deep`

## Configuration

### Environment Variables

Set in `.env.local`:

```bash
# Required (at least one)
EXA_API_KEY=...           # Exa web search
PERPLEXITY_API_KEY=...    # Perplexity AI search

# Optional
CONTEXT7_API_KEY=...      # Higher rate limits for Context7
KAGI_API_KEY=...          # Kagi search
SEARXNG_URL=...           # Self-hosted SearXNG instance
NEO4J_URI=...             # CodeGraphContext
NEO4J_USERNAME=...
NEO4J_PASSWORD=...

# Feature flags
ENCYCLOPEDIA_ENABLE_CONTEXT7=true
ENCYCLOPEDIA_ENABLE_KAGI=false
ENCYCLOPEDIA_ENABLE_SEARXNG=false
ENCYCLOPEDIA_ENABLE_CODEGRAPH=false
```

### Source Routing

| Query Type | Primary Sources | Fallback |
|------------|-----------------|----------|
| `library_docs` | context7 | exa |
| `general_search` | exa, perplexity | kagi, searxng |
| `code_context` | mcp_git_ingest | exa, codegraph |

**Query Classification:**
1. Explicit hints: `doc:`, `code:`, `web:` prefixes
2. Repository hints: `repo:owner/name` triggers code context
3. Pattern matching: URLs → general search; `def/class` → code context
4. Keywords: `latest`, `2024` → general search
5. Default: library docs

## Output Format

```json
{
  "status": "success",
  "query_type": "library_docs",
  "results": [
    {
      "title": "Result title",
      "content": "Result content...",
      "url": "https://...",
      "source": "exa",
      "relevance": 0.85
    }
  ],
  "sources_used": ["context7", "exa"],
  "degraded": false,
  "degradation": {"missing": [], "errors": []}
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 1 | Configuration error (missing credentials) |
| 2 | No results found |
| 3 | Backend unavailable |
| 4 | Internal error |

## Directory Structure

```
encyclopedia/
├── SKILL.md              # Skill metadata for Claude
├── README.md             # This file
├── .env.local            # Symlink to project env
├── resources/
│   └── source_config.json
└── scripts/
    ├── encyclopedia.py   # Main entrypoint
    ├── context7client/   # Context7 API client
    ├── exaclient/        # Exa AI client
    ├── perplexity/       # Perplexity client
    ├── kagiclient/       # Kagi client
    ├── searxng/          # SearXNG client
    ├── gitingest/        # Git ingest client
    ├── codegraph/        # CodeGraph client
    └── inmemoria/        # In-memory storage
```

## Dependencies

- Python >= 3.10
- httpx

## License

MIT

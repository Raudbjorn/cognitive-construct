---
name: anime-mori
description: >-
  Persistent codebase intelligence. Learns patterns, conventions, and architecture
  from your codebase via tree-sitter AST parsing (12 languages) and remembers across
  sessions. MCP server with 13 tools for semantic search, pattern prediction, file
  routing, and project blueprints.
license: MIT
metadata:
  version: 0.6.0
  runtime: node>=18
  dependencies:
    required:
      - "@modelcontextprotocol/sdk"
      - "@surrealdb/node"
      - tree-sitter (via rust-core NAPI bindings)
    optional:
      - "@xenova/transformers"
  env_vars:
    required: []
    optional:
      - ANIME_MORI_DB_PATH
---

# Anime Mori: Persistent Codebase Intelligence

> "Your AI pair programmer finally has a notepad that doesn't get wiped clean."

## Overview

**Anime Mori** is the codebase memory skill of the Cognitive Construct. It solves session amnesia — the problem where AI assistants forget your architecture, naming conventions, and design decisions every time you start a new session.

Built on Rust (tree-sitter AST parsing) + TypeScript (MCP server + orchestration), it learns once and remembers forever via SurrealDB persistence.

## Architecture

```text
AI Tool (Claude/Copilot)
  │ MCP Protocol (stdio)
  ▼
TypeScript Server — 13 tools, 4 categories
  │ NAPI bindings
  ▼
Rust Core — tree-sitter, pattern learning, semantic engine
  │
  ▼
Storage — SurrealDB (persistent vectors) + SQLite (structured data)
```

## Commands

### `learn <path>`
Analyze and learn from a codebase. Extracts patterns, architecture, and conventions.

```bash
npx anime-mori learn ./my-project
```

### `server`
Start the MCP server for AI tool integration.

```bash
npx anime-mori server
```

### `search <query>`
Multi-mode codebase search (semantic, text, or pattern-based).

```bash
npx anime-mori search "authentication middleware"
```

### `analyze <path>`
Analyze files/directories with concepts, patterns, and complexity metrics.

### `insights <query>`
Query learned semantic concepts and relationships.

### `patterns`
Get pattern recommendations with examples from your codebase.

### `predict`
Predict implementation approach based on your coding history.

### `status`
System health check.

## MCP Tools (13)

| Category | Tools | Description |
|----------|-------|-------------|
| Core Analysis (2) | `analyze_codebase`, `search_codebase` | File/directory analysis, multi-mode search |
| Intelligence (7) | `learn_codebase_intelligence`, `get_project_blueprint`, `get_semantic_insights`, `get_pattern_recommendations`, `predict_coding_approach`, `get_developer_profile`, `contribute_insights` | Deep learning, blueprints, predictions |
| Automation (1) | `auto_learn_if_needed` | Smart auto-learning with staleness detection |
| Monitoring (3) | `get_system_status`, `get_intelligence_metrics`, `get_performance_status` | Health, analytics, diagnostics |

## Language Support

Native AST parsing via tree-sitter (12 languages):

TypeScript, JavaScript (JSX/TSX), Python, Rust, Go, Java, C, C++, C#, Svelte, SQL, PHP

## Synergies

- **→ Encyclopedia**: Vendored at `encyclopedia/scripts/anime-mori/` — encyclopedia can invoke Anime Mori's MCP tools for codebase context during knowledge retrieval
- **→ Inland Empire**: Learned patterns and conventions feed into the subconscious memory layer

## Files

- `src/` — TypeScript source (CLI, MCP server, engines, storage)
- `rust-core/` — Rust NAPI bindings (tree-sitter parsers, analyzers)
- `src/mcp-server/server.ts` — MCP server entry point
- `src/cli/cli.ts` — CLI entry point
- `src/engines/` — Semantic and pattern analysis engines
- `src/storage/` — SurrealDB persistence layer
- `README.md` — Full documentation with setup guides

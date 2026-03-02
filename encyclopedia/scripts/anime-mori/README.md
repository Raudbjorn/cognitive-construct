# Anime Mori

---

## The Problem: Session Amnesia

You know the drill. You fire up Claude, Copilot, or Cursor to help with your codebase. You explain your architecture. You describe your patterns. You outline your conventions. The AI gets it, helps you out, and everything's great.

Then you close the window.

Next session? Complete amnesia. You're explaining the same architectural decisions again. The same naming conventions. The same "no, we don't use classes here, we use functional composition" for the fifteenth time.

**Every AI coding session starts from scratch.**

This isn't just annoying, it's inefficient. These tools re-analyze your codebase on every interaction, burning tokens and time. They give generic suggestions that don't match your style. They have no memory of what worked last time, what you rejected, or why.

## The Solution: Persistent Intelligence

Anime Mori is an MCP server that learns from your actual codebase and remembers across sessions. It builds persistent intelligence about your code (patterns, architecture, conventions, decisions) that AI assistants can query through the Model Context Protocol.

Think of it as giving your AI pair programmer a notepad that doesn't get wiped clean every time you restart the session.

**Current version: 0.6.0** - [See what's changed](CHANGELOG.md)

### What It Does

- **Learns your patterns** - Analyzes your code to understand naming conventions, architectural choices, and structural preferences
- **Instant project context** - Provides tech stack, entry points, and architecture in <200 tokens (no re-analysis needed)
- **Smart file routing** - Routes vague requests like "add password reset" directly to relevant files
- **Semantic search** - Finds code by meaning, not just keywords
- **Work memory** - Tracks current tasks and architectural decisions across sessions
- **Pattern prediction** - Suggests how you'd solve similar problems based on your history

### Example Workflow

```bash
# First time: Learn your codebase
npx anime-mori learn ./my-project

# Start the MCP server
npx anime-mori server

# Now in Claude/Copilot:
You: "Add password reset functionality"
AI: *queries Anime Mori*
    "Based on your auth patterns in src/auth/login.ts, I'll use your
     established JWT middleware pattern and follow your Result<T>
     error handling convention..."

# Next session (days later):
You: "Where did we put the password reset code?"
AI: *queries Anime Mori*
    "In src/auth/password-reset.ts, following the pattern we
     established in our last session..."
```

No re-explaining. No generic suggestions. Just continuous, context-aware assistance.

## Quick Start

### Installation

```bash
# Install globally
npm install -g anime-mori

# Or use directly with npx
npx anime-mori --help
```

### Connect to Your AI Tool

**Claude Desktop** - Add to your config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "anime-mori": {
      "command": "npx",
      "args": ["anime-mori", "server"]
    }
  }
}
```

**Claude Code CLI**:

```bash
claude mcp add anime-mori -- npx anime-mori server
```

**GitHub Copilot** - See [Copilot Integration](#github-copilot-integration) section below

### Learn Your Codebase

```bash
# Analyze and learn from your project
npx anime-mori learn ./my-project

# Or let AI agents trigger learning automatically
# (Just start the server and let auto_learn_if_needed handle it)
npx anime-mori server
```

## How It Works

Anime Mori is built on Rust + TypeScript, using the Model Context Protocol to connect AI tools to persistent codebase intelligence.

### Architecture

```
┌─────────────────────┐    MCP     ┌──────────────────────┐    napi-rs    ┌─────────────────────┐
│  AI Tool (Claude)   │◄──────────►│  TypeScript Server   │◄─────────────►│   Rust Core         │
└─────────────────────┘            └──────────┬───────────┘               │  • AST Parser       │
                                              │                           │  • Pattern Learner  │
                                              │                           │  • Semantic Engine  │
                                              ▼                           │  • Blueprint System │
                                   ┌──────────────────────┐               └─────────────────────┘
                                   │ SQLite (persistent)  │
                                   │ SurrealDB (in-mem)   │
                                   └──────────────────────┘
```

### The Core Components

**Rust Layer** - Fast, native processing:

- Tree-sitter AST parsing for 12 languages (TypeScript, JavaScript, Python, PHP, Rust, Go, Java, C/C++, C#, Svelte, SQL)
- Blueprint analyzer (detects project structure, entry points, architecture patterns)
- Pattern learner (statistical analysis of your coding style)
- Semantic engine (understands code relationships and concepts)

**TypeScript Layer** - MCP server and orchestration:

- 13 specialized tools for AI assistants (organized into 4 categories)
- SQLite for structured data, SurrealDB with SurrealKV for persistent vector embeddings
- File watching for incremental updates
- Smart routing that maps features to files

**Storage** - Local-first:

- Everything stays on your machine
- SQLite for patterns and metadata
- SurrealDB with SurrealKV backend for persistent vector embeddings
- Local transformers.js for embeddings (Xenova/all-MiniLM-L6-v2)

### What Makes It Different

This isn't just another RAG system or static rules engine:

- **Learns from actual code** - Not manually-defined rules, but statistical patterns from your real codebase
- **Predicts your approach** - Based on how you've solved similar problems before
- **Token efficient** - Responses optimized to minimize LLM context usage (<200 tokens for project context)
- **Routes to files** - "Add login" → automatically suggests `src/auth/login.ts`
- **Remembers context** - Tracks work sessions, tasks, and architectural decisions
- **Multi-mode search** - Semantic (meaning), text (keywords), or pattern-based

## What's New in v0.5.x

We recently completed Phases 1-4 of the implementation roadmap:

### 🗺️ Project Blueprints (Phase 1)

Instant project context without full learning. Ask about a codebase and get tech stack, entry points, key directories, and architecture all in under 200 tokens.

### 💼 Work Context System (Phase 2)

AI agents can now track work sessions, maintain task lists, and record architectural decisions. Resume work exactly where you left off.

### 🧭 Smart File Routing (Phase 3)

Feature-to-file mapping across 10 categories (auth, API, database, UI, etc.). Vague requests like "add password reset" get routed to specific files automatically.

### ⚡ Smooth Progress Tracking (v0.5.3)

No more janky console spam. Progress bars update in-place with consistent 500ms refresh rates.

## MCP Tools for AI Assistants

Anime Mori provides **13 specialized tools** that AI assistants can call via MCP. They're organized into 4 categories (down from 16 after Phase 4 consolidation merged redundant tools):

### 🎯 Core Analysis (2 tools)

- `analyze_codebase` - Analyze files/directories with concepts, patterns, complexity (Phase 4: now handles both files and directories)
- `search_codebase` - Multi-mode search (semantic/text/pattern)

### 🧠 Intelligence (7 tools)

- `learn_codebase_intelligence` - Deep learning to extract patterns and architecture
- `get_project_blueprint` - Instant project context with tech stack and entry points ⭐ (Phase 4: includes learning status)
- `get_semantic_insights` - Query learned concepts and relationships
- `get_pattern_recommendations` - Get patterns with related files for consistency
- `predict_coding_approach` - Implementation guidance with file routing ⭐
- `get_developer_profile` - Access coding style and work context
- `contribute_insights` - Record architectural decisions

### 🤖 Automation (1 tool)

- `auto_learn_if_needed` - Smart auto-learning with staleness detection ⭐ (Phase 4: includes quick setup functionality)

### 📊 Monitoring (3 tools)

- `get_system_status` - Health check
- `get_intelligence_metrics` - Analytics on learned patterns
- `get_performance_status` - Performance diagnostics

**Phase 4 Consolidation**: Three tools were merged into existing tools for better AX (agent experience haha):

- ~~get_file_content~~ → merged into `analyze_codebase`
- ~~get_learning_status~~ → merged into `get_project_blueprint`
- ~~quick_setup~~ → merged into `auto_learn_if_needed`

> **For AI agents**: See [`AGENT.md`](AGENT.md) for complete tool reference with usage patterns and decision trees.

## GitHub Copilot Integration

Anime Mori works with GitHub Copilot through custom instructions and chat modes.

### Setup

This repository includes:

- `.github/copilot-instructions.md` - Automatic guidance for Copilot
- `.github/chatmodes/` - Three specialized chat modes:
  - 🔍 **animemori-explorer** - Intelligent codebase navigation
  - 🚀 **animemori-feature** - Feature implementation with patterns
  - 🔎 **animemori-review** - Code review with consistency checking

## GitHub Copilot Integration (VS Code)

Anime Mori integrates with **GitHub Copilot Chat** using **MCP + Custom Agents** (formerly called *Chat Modes*).
These agents allow Copilot to query Anime Mori’s persistent intelligence when working in **Agent mode**.

> ⚠️ **Important**: Copilot will only call MCP tools when the chat is in **Agent** mode (not Ask or Edit).

### Requirements

* **VS Code 1.106+**
* **GitHub Copilot Chat** enabled
* Anime Mori MCP server configured and running

---

### MCP Server Setup (Required)

Create or edit the following file in your workspace:

**`.vscode/mcp.json`**

```json
{
  "servers": {
    "anime-mori": {
      "command": "npx",
      "args": ["anime-mori", "server"]
    }
  }
}
```

Open this file in VS Code and click **Start** when prompted, or start it manually.

---

### Copilot Instructions (Automatic)

This repository includes:

* **`.github/copilot-instructions.md`**

VS Code automatically loads this file and applies guidance to Copilot Chat.
No additional setup is required.

---

### Custom Agents (formerly “Chat Modes”)

This repository provides **three Custom Agents** for Copilot:

| Agent                     | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| 🔍 **animemori-explorer** | Intelligent codebase navigation               |
| 🚀 **animemori-feature**  | Feature implementation using learned patterns |
| 🔎 **animemori-review**   | Consistency & pattern-based code review       |

#### Agent File Location

> ⚠️ VS Code has renamed *Chat Modes* → **Custom Agents**

To ensure compatibility with current VS Code versions:

1. Create the folder:

   ```
   .github/agents/
   ```
2. Move or copy files from:

   ```
   .github/chatmodes/
   ```

   into:

   ```
   .github/agents/
   ```
3. Rename each file:

   ```
   *.chatmode.md → *.agent.md
   ```

Example:

```
animemori-feature.chatmode.md → animemori-feature.agent.md
```

---

### Using Agents in VS Code

1. Open **Copilot Chat**
2. Select the custom agent (e.g., animemori-featurename) directly from the dropdown.

If agents do not appear:

* Reload the window
* Ensure files are in `.github/agents/`

---

### Example Usage

```text
Where is the authentication logic?
```

→ Copilot queries Anime Mori’s semantic index

```text
Add password reset functionality
```

→ Copilot retrieves:

* file routing
* architectural patterns
* prior auth decisions

```text
Review this code for consistency
```

→ Copilot compares against learned conventions

---

### Common Pitfalls

* ❌ Using **Ask/Edit mode** → MCP tools are ignored
* ❌ MCP server not running
* ❌ Agent files left in `.github/chatmodes/` only
* ❌ VS Code version < 1.106

---

### Notes on VS Code Terminology

| Old Name                      | Current Name                    |
| ----------------------------- | ------------------------------- |
| Chat Modes                    | Custom Agents                   |
| `Chat: Configure Chat Modes…` | `Chat: Configure Custom Agents` |
| `.github/chatmodes/`          | `.github/agents/`               |

VS Code still recognizes legacy files, but **`.github/agents/*.agent.md` is the recommended format going forward**.

## Language Support

Native AST parsing via tree-sitter for:

- TypeScript & JavaScript (including JSX/TSX)
- Python
- PHP
- Rust
- Go
- Java
- C & C++
- C#
- Svelte
- SQL

Build artifacts (`node_modules/`, `dist/`, `.next/`, etc.) are automatically filtered out.

## Status: Work in Progress

**Let's be honest**: Anime Mori is early-stage software. It works, but it's not perfect.

### What Works Well

- ✅ Pattern learning from real codebases
- ✅ Semantic search across concepts
- ✅ Project blueprint generation
- ✅ MCP integration with Claude Desktop/Code
- ✅ Cross-platform support (Linux, macOS, Windows)
- ✅ Token-efficient responses

### Known Limitations

- ⚠️ Large codebases (100k+ files) can be slow on first analysis
- ⚠️ Pattern accuracy improves with codebase consistency
- ⚠️ Some languages have better tree-sitter support than others
- ⚠️ Documentation could be more comprehensive


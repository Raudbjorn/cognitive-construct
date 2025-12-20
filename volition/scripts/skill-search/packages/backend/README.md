# Claude Skills MCP Backend

Heavy backend server for Claude Skills MCP system with vector search capabilities.

## Overview

This is the backend component of the Claude Skills MCP system. It provides:
- Vector-based semantic search using sentence-transformers
- Skill indexing and retrieval
- MCP protocol via Streamable HTTP transport
- Background skill loading from GitHub and local sources

**Note**: This package is typically auto-installed by the frontend (`skill-search`). You only need to install it manually for:
- Remote deployment (hosting your own backend)
- Development and testing
- Standalone usage without the frontend proxy

## Installation

```bash
# Via uv (recommended)
uv tool install skill-search-backend

# Via uvx (one-time use)
uvx skill-search-backend

# Via pip
pip install skill-search-backend
```

## Usage

### Run Standalone Server

```bash
# Default (localhost:8765)
skill-search-backend

# Custom port
skill-search-backend --port 8080

# For remote access
skill-search-backend --host 0.0.0.0 --port 8080

# With custom configuration
skill-search-backend --config my-config.json

# Verbose logging
skill-search-backend --verbose
```

### Configuration

```bash
# Print example configuration
skill-search-backend --example-config > config.json

# Edit config.json to customize skill sources, embedding model, etc.

# Run with custom config
skill-search-backend --config config.json
```

## Endpoints

When running, the backend exposes:

- **Streamable HTTP MCP**: `http://localhost:8765/mcp`
- **Health Check**: `http://localhost:8765/health`

## Docker Deployment

### Build Image

```bash
docker build -t skill-search-backend .
```

### Run Container

```bash
# For local access
docker run -p 8765:8765 skill-search-backend

# For remote access
docker run -p 8080:8765 \
  -e HOST=0.0.0.0 \
  skill-search-backend --host 0.0.0.0 --port 8765
```

## Dependencies

This package includes heavy dependencies (~250 MB):
- PyTorch (CPU-only on Linux): ~150-200 MB
- sentence-transformers: ~50 MB
- numpy, httpx, fastapi, uvicorn: ~30 MB

**First download may take 60-180 seconds** depending on your internet connection.

## Performance

- **Startup time**: 2-5 seconds (with cached dependencies)
- **First search**: +2-5 seconds (embedding model download, one-time)
- **Query time**: <1 second after models loaded
- **Memory usage**: ~500 MB

## Development

```bash
# Clone the monorepo
git clone https://github.com/K-Dense-AI/skill-search.git
cd skill-search/packages/backend

# Install in development mode
uv pip install -e ".[test]"

# Run tests
uv run pytest tests/
```

## Related Packages

- **skill-search** (Frontend): Lightweight proxy that auto-installs this backend
- **Main Repository**: https://github.com/K-Dense-AI/skill-search

## License

Apache License 2.0

Copyright 2025 K-Dense AI (https://k-dense.ai)


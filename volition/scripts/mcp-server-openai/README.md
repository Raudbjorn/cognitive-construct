# OpenAI MCP Server

Query OpenAI models directly from Claude using MCP protocol.

![preview](preview.png)

## Setup

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openai-server": {
      "command": "mcp-server-openai",
      "args": ["serve"],
      "env": { "OPENAI_API_KEY": "your-key-here" }
    }
  }
}
```

## Development
```bash
git clone https://github.com/pierrebrunelle/mcp-server-openai
cd mcp-server-openai
pip install -e .
```

## CLI
```bash
# Start the MCP stdio server
mcp-server-openai serve

# Ask OpenAI directly (prints to stdout)
mcp-server-openai ask "Hello, how are you?"

# From source (after `pip install -e .`)
python -m mcp_server_openai.cli serve
```

## Testing
```bash
pytest -q
```

## License
MIT License

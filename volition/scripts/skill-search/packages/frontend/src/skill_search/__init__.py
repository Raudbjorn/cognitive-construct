"""Claude Skills MCP Frontend - Lightweight proxy server."""

from __future__ import annotations

__version__ = "1.0.6"

# Result types
from .result import Err, Ok, Result

# Backend management
from .backend_manager import BackendError, BackendManager

# MCP proxy
from .mcp_proxy import MCPProxy, TOOL_SCHEMAS

__all__ = [
    # Version
    "__version__",
    # Result types
    "Ok",
    "Err",
    "Result",
    # Backend management
    "BackendError",
    "BackendManager",
    # MCP proxy
    "MCPProxy",
    "TOOL_SCHEMAS",
]

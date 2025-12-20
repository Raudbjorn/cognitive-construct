"""Shared tool schemas for Claude Skills MCP.

This module is the single source of truth for tool definitions.
Both frontend (proxy) and backend use these exact schemas to avoid mismatches
that cause "invalid tool use" errors in Claude.

IMPORTANT: If you modify these schemas, ensure both packages are updated/released.
"""

from __future__ import annotations

from mcp.types import Tool

# Constants - these MUST match between frontend and backend
DEFAULT_TOP_K = 3
MAX_TOP_K = 20
MIN_TOP_K = 1


def get_tool_schemas() -> list[Tool]:
    """Return the canonical tool schemas.

    These schemas are used by both the frontend proxy (for instant response)
    and the backend server (for validation). They MUST be identical.

    Returns
    -------
    list[Tool]
        List of MCP Tool definitions.
    """
    return [
        Tool(
            name="find_helpful_skills",
            title="Find the most helpful skill for any task",
            description=(
                "Always call this tool FIRST whenever the question requires any "
                "domain-specific knowledge beyond common sense or simple recall. "
                "Use it at task start, regardless of the task and whether you are "
                "sure about the task, It performs semantic search over a curated "
                "library of proven skills and returns ranked candidates with "
                "step-by-step guidance and best practices. Do this before any "
                "searches, coding, or any other actions as this will inform you "
                "about the best approach to take."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Description of the task you want to accomplish. "
                            "Be specific about your goal, context, or problem domain "
                            "for better results (e.g., 'debug Python API errors', "
                            "'process genomic data', 'build React dashboard')"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            f"Number of skills to return (default: {DEFAULT_TOP_K}). "
                            "Higher values provide more options but may include less "
                            "relevant results."
                        ),
                        "default": DEFAULT_TOP_K,
                        "minimum": MIN_TOP_K,
                        "maximum": MAX_TOP_K,
                    },
                    "list_documents": {
                        "type": "boolean",
                        "description": (
                            "Include a list of available documents (scripts, "
                            "references, assets) for each skill (default: True)"
                        ),
                        "default": True,
                    },
                },
                "required": ["task_description"],
            },
        ),
        Tool(
            name="read_skill_document",
            title="Open skill documents and assets",
            description=(
                "Use after finding a relevant skill to retrieve specific documents "
                "(scripts, references, assets). Supports pattern matching "
                "(e.g., 'scripts/*.py') to fetch multiple files. Returns text content "
                "or URLs and never executes code. Prefer pulling only the files you "
                "need to complete the current step."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": (
                            "Name of the skill (as returned by find_helpful_skills)"
                        ),
                    },
                    "document_path": {
                        "type": "string",
                        "description": (
                            "Path or pattern to match documents. Examples: "
                            "'scripts/example.py', 'scripts/*.py', 'references/*', "
                            "'assets/diagram.png'. If not provided, returns a list "
                            "of all available documents."
                        ),
                    },
                    "include_base64": {
                        "type": "boolean",
                        "description": (
                            "For images: if True, return base64-encoded content; "
                            "if False, return only URL. Default: False (URL only "
                            "for efficiency)"
                        ),
                        "default": False,
                    },
                },
                "required": ["skill_name"],
            },
        ),
        Tool(
            name="list_skills",
            title="List available skills",
            description=(
                "Returns the full inventory of loaded skills (names, descriptions, "
                "sources, document counts) for exploration or debugging. For task-driven "
                "work, prefer calling 'find_helpful_skills' first to locate the most "
                "relevant option before reading documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# Pre-computed for instant access (used by frontend proxy)
TOOL_SCHEMAS = get_tool_schemas()

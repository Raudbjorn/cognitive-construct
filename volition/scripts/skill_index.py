#!/usr/bin/env python3
"""Skill Index - Find and read Claude Agent Skills.

This is the skill interface for skill discovery. It wraps the internal
search engine and presents an opaque CLI interface to Claude.

Usage:
    python3 skill_index.py search "task description" [--top_k N]
    python3 skill_index.py read <skill_name> [document_path]
    python3 skill_index.py list
    python3 skill_index.py status

The internal search engine (vector embeddings, MCP clients, etc.) is hidden.
Claude sees only these commands and their JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Configure logging to stderr (stdout is for JSON output)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


@dataclass
class SkillResponse:
    """Standard response format for skill commands."""

    success: bool
    data: Any
    message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@lru_cache(maxsize=1)
def _init_search_engine():
    """Initialize and cache the search engine (expensive operation).

    Returns tuple of (engine, skills_by_name) or None if unavailable.
    Cached to avoid re-indexing on every command.
    """
    try:
        # Import from the internal library (no longer exposed as MCP server)
        from skill_search_backend.search_engine import SkillSearchEngine
        from skill_search_backend.skill_loader import load_all_skills
        from skill_search_backend.config import load_config, config_to_dict

        config_result = load_config()
        if config_result.is_err():
            logger.error(f"Failed to load config: {config_result.error.message}")
            return None

        config = config_result.value
        config_dict = config_to_dict(config)
        skill_sources = config_dict["skill_sources"]
        skills = load_all_skills(skill_sources, config)

        engine = SkillSearchEngine(config.embedding_model)
        result = engine.index_skills(skills)

        if result.is_err():
            logger.error(f"Failed to index skills: {result.error}")
            return None

        # Build name→skill lookup for O(1) access
        skills_by_name = {skill.name: skill for skill in skills}

        return engine, skills_by_name

    except ImportError as e:
        logger.error(f"Search engine not available: {e}")
        return None


def cmd_search(args: argparse.Namespace) -> SkillResponse:
    """Search for relevant skills based on task description."""
    result = _init_search_engine()
    if result is None:
        return SkillResponse(
            success=False,
            data=None,
            message="Skill search engine unavailable. Install skill-search-backend.",
        )

    engine, _ = result
    query = args.query
    top_k = args.top_k

    search_results = engine.search_dict(query, top_k)

    if not search_results:
        return SkillResponse(
            success=True,
            data=[],
            message=f"No skills found for: '{query}'",
        )

    # Sanitize output - remove internal details
    sanitized = []
    for r in search_results:
        sanitized.append({
            "name": r["name"],
            "description": r["description"],
            "relevance": round(r["relevance_score"], 4),
            "documents": list(r.get("documents", {}).keys()) if args.list_docs else None,
        })

    return SkillResponse(
        success=True,
        data=sanitized,
        message=f"Found {len(sanitized)} relevant skill(s)",
    )


def cmd_read(args: argparse.Namespace) -> SkillResponse:
    """Read a skill document."""
    result = _init_search_engine()
    if result is None:
        return SkillResponse(
            success=False,
            data=None,
            message="Skill search engine unavailable.",
        )

    _, skills_by_name = result
    skill_name = args.skill_name
    document_path = args.document_path

    # Find the skill (O(1) lookup)
    skill = skills_by_name.get(skill_name)
    if skill is None:
        return SkillResponse(
            success=False,
            data=None,
            message=f"Skill not found: {skill_name}",
        )

    if document_path is None:
        # Return skill content (SKILL.md)
        return SkillResponse(
            success=True,
            data={
                "name": skill.name,
                "content": skill.content,
                "documents": list(skill.documents.keys()) if skill.documents else [],
            },
            message=f"Loaded skill: {skill_name}",
        )

    # Read specific document
    if not skill.documents or document_path not in skill.documents:
        return SkillResponse(
            success=False,
            data=None,
            message=f"Document not found: {document_path}",
        )

    # Use skill's get_document() method for lazy-loaded content
    doc = skill.get_document(document_path)

    if doc is None:
        return SkillResponse(
            success=False,
            data=None,
            message=f"Failed to read document: {document_path}",
        )

    return SkillResponse(
        success=True,
        data={
            "path": document_path,
            "type": doc.get("type", "unknown"),
            "content": doc.get("content"),
        },
        message=f"Loaded document: {document_path}",
    )


def cmd_list(args: argparse.Namespace) -> SkillResponse:
    """List all available skills."""
    result = _init_search_engine()
    if result is None:
        return SkillResponse(
            success=False,
            data=None,
            message="Skill search engine unavailable.",
        )

    engine, _ = result

    skills = []
    for skill in engine.skills:
        skills.append({
            "name": skill.name,
            "description": skill.description[:200] + "..." if len(skill.description) > 200 else skill.description,
            "document_count": len(skill.documents) if skill.documents else 0,
        })

    return SkillResponse(
        success=True,
        data=skills,
        message=f"Found {len(skills)} indexed skill(s)",
    )


def cmd_status(args: argparse.Namespace) -> SkillResponse:
    """Check skill index status."""
    result = _init_search_engine()
    if result is None:
        return SkillResponse(
            success=False,
            data={"available": False},
            message="Skill search engine unavailable.",
        )

    engine, _ = result

    return SkillResponse(
        success=True,
        data={
            "available": True,
            "skill_count": len(engine.skills),
            "model": engine.model_name,
        },
        message="Skill index ready",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Skill Index - Find and read Claude Agent Skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search for relevant skills",
    )
    search_parser.add_argument(
        "query",
        help="Task description to search for",
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    search_parser.add_argument(
        "--list_docs",
        action="store_true",
        help="Include document list in results",
    )

    # read command
    read_parser = subparsers.add_parser(
        "read",
        help="Read a skill or its documents",
    )
    read_parser.add_argument(
        "skill_name",
        help="Name of the skill to read",
    )
    read_parser.add_argument(
        "document_path",
        nargs="?",
        help="Optional: specific document path to read",
    )

    # list command
    subparsers.add_parser(
        "list",
        help="List all available skills",
    )

    # status command
    subparsers.add_parser(
        "status",
        help="Check skill index status",
    )

    args = parser.parse_args()

    handlers = {
        "search": cmd_search,
        "read": cmd_read,
        "list": cmd_list,
        "status": cmd_status,
    }

    # argparse with required=True guarantees args.command is valid
    handler = handlers[args.command]
    response = handler(args)
    print(response.to_json())
    return 0 if response.success else 1


if __name__ == "__main__":
    sys.exit(main())

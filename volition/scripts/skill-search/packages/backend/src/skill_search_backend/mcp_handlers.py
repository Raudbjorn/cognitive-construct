"""MCP server implementation for Claude Skills search."""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .search_engine import SkillSearchEngine
from .tool_schemas import DEFAULT_TOP_K, TOOL_SCHEMAS
from .types import LoadingState

logger = logging.getLogger(__name__)


class SkillsMCPServer:
    """MCP Server for searching Claude Agent Skills.

    Attributes
    ----------
    search_engine : SkillSearchEngine
        The search engine instance.
    default_top_k : int
        Default number of results to return.
    max_content_chars : int | None
        Maximum characters for skill content (None for unlimited).
    loading_state : LoadingState
        State tracker for background skill loading.
    """

    def __init__(
        self,
        search_engine: SkillSearchEngine,
        loading_state: LoadingState,
        default_top_k: int = DEFAULT_TOP_K,
        max_content_chars: int | None = None,
    ):
        """Initialize the MCP server.

        Parameters
        ----------
        search_engine : SkillSearchEngine
            Initialized search engine with indexed skills.
        loading_state : LoadingState
            State tracker for background skill loading.
        default_top_k : int, optional
            Default number of results to return, by default 3.
        max_content_chars : int | None, optional
            Maximum characters for skill content. None for unlimited, by default None.
        """
        self.search_engine = search_engine
        self.loading_state = loading_state
        self.default_top_k = default_top_k
        self.max_content_chars = max_content_chars
        self.server = Server("skill-search")

        self._register_handlers()

        logger.info("MCP server initialized")

    def _register_handlers(self) -> None:
        """Register MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools.

            Returns the shared TOOL_SCHEMAS to ensure frontend/backend consistency.
            """
            return TOOL_SCHEMAS

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            if name == "find_helpful_skills":
                return await self._handle_search_skills(arguments)
            elif name == "read_skill_document":
                return await self._handle_read_skill_document(arguments)
            elif name == "list_skills":
                return await self._handle_list_skills(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_search_skills(
        self, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle find_helpful_skills tool calls."""
        task_description = arguments.get("task_description")
        if not task_description:
            raise ValueError("task_description is required")

        top_k = arguments.get("top_k", self.default_top_k)
        list_documents = arguments.get("list_documents", True)

        response_parts: list[str] = []

        status_msg = self.loading_state.get_status_message()
        if status_msg:
            response_parts.append(status_msg)

        # Use search_dict for backward compatibility
        results = self.search_engine.search_dict(task_description, top_k)

        if not results:
            if (
                not self.loading_state.is_complete
                and self.loading_state.loaded_skills == 0
            ):
                return [
                    TextContent(
                        type="text",
                        text=(status_msg or "")
                        + "No skills loaded yet. Please wait for skills to load and try again.",
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text="No relevant skills found for the given task description.",
                )
            ]

        response_parts.append(
            f"Found {len(results)} relevant skill(s) for: '{task_description}'\n"
        )

        for i, result in enumerate(results, 1):
            response_parts.append(f"\n{'=' * 80}")
            response_parts.append(f"\nSkill {i}: {result['name']}")
            response_parts.append(f"\nRelevance Score: {result['relevance_score']:.4f}")
            response_parts.append(f"\nSource: {result['source']}")
            response_parts.append(f"\nDescription: {result['description']}")

            documents = result.get("documents", {})
            if documents:
                response_parts.append(f"\nAdditional Documents: {len(documents)} file(s)")

                if list_documents:
                    response_parts.append("\nAvailable Documents:")
                    for doc_path in sorted(documents.keys()):
                        doc_info = documents[doc_path]
                        doc_type = doc_info.get("type", "unknown")
                        doc_size = doc_info.get("size", 0)
                        size_kb = doc_size / 1024
                        response_parts.append(
                            f"  - {doc_path} ({doc_type}, {size_kb:.1f} KB)"
                        )

            response_parts.append(f"\n{'-' * 80}")
            response_parts.append("\nFull Content:\n")

            content = result["content"]
            if (
                self.max_content_chars is not None
                and len(content) > self.max_content_chars
            ):
                truncated_content = content[: self.max_content_chars] + "..."
                response_parts.append(truncated_content)
                response_parts.append(
                    f"\n\n[Content truncated at {self.max_content_chars} characters. "
                    f"View full skill at: {result['source']}]"
                )
            else:
                response_parts.append(content)

            response_parts.append(f"\n{'=' * 80}\n")

        return [TextContent(type="text", text="\n".join(response_parts))]

    async def _handle_read_skill_document(
        self, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle read_skill_document tool calls."""
        skill_name = arguments.get("skill_name")
        if not skill_name:
            raise ValueError("skill_name is required")

        document_path = arguments.get("document_path")
        include_base64 = arguments.get("include_base64", False)

        skill = None
        for s in self.search_engine.skills:
            if s.name == skill_name:
                skill = s
                break

        if not skill:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Skill '{skill_name}' not found. Please use find_helpful_skills "
                        "to find valid skill names."
                    ),
                )
            ]

        if not document_path:
            if not skill.documents:
                return [
                    TextContent(
                        type="text",
                        text=f"Skill '{skill_name}' has no additional documents.",
                    )
                ]

            response_parts = [f"Available documents for skill '{skill_name}':\n"]
            for doc_path, doc_info in sorted(skill.documents.items()):
                doc_type = doc_info.get("type", "unknown")
                doc_size = doc_info.get("size", 0)
                size_kb = doc_size / 1024
                response_parts.append(f"  - {doc_path} ({doc_type}, {size_kb:.1f} KB)")

            return [TextContent(type="text", text="\n".join(response_parts))]

        matching_docs: dict[str, dict[str, Any]] = {}
        for doc_path, doc_info in skill.documents.items():
            if fnmatch.fnmatch(doc_path, document_path) or doc_path == document_path:
                matching_docs[doc_path] = doc_info

        if not matching_docs:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"No documents matching '{document_path}' found in "
                        f"skill '{skill_name}'."
                    ),
                )
            ]

        for doc_path in matching_docs:
            doc_info = matching_docs[doc_path]
            if not doc_info.get("fetched") and "content" not in doc_info:
                content = skill.get_document(doc_path)
                if content:
                    matching_docs[doc_path] = content

        response_parts: list[str] = []

        if len(matching_docs) == 1:
            doc_path, doc_info = list(matching_docs.items())[0]
            doc_type = doc_info.get("type")

            if doc_type == "text":
                response_parts.append(f"Document: {doc_path}\n")
                response_parts.append("=" * 80)
                response_parts.append(f"\n{doc_info.get('content', '')}")

            elif doc_type == "image":
                response_parts.append(f"Image: {doc_path}\n")
                if doc_info.get("size_exceeded"):
                    response_parts.append(
                        f"Size: {doc_info.get('size', 0) / 1024:.1f} KB (exceeds limit)"
                    )
                    response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")
                elif include_base64:
                    response_parts.append(
                        f"Base64 Content:\n{doc_info.get('content', '')}"
                    )
                    if "url" in doc_info:
                        response_parts.append(
                            f"\n\nAlternatively, access via URL: {doc_info['url']}"
                        )
                else:
                    response_parts.append(f"URL: {doc_info.get('url', 'N/A')}")
                    if "content" in doc_info:
                        response_parts.append(
                            "\n(Set include_base64=true to get base64-encoded content)"
                        )

        else:
            response_parts.append(
                f"Found {len(matching_docs)} documents matching '{document_path}':\n"
            )

            for doc_path, doc_info in sorted(matching_docs.items()):
                doc_type = doc_info.get("type")
                response_parts.append(f"\n{'=' * 80}")
                response_parts.append(f"\nDocument: {doc_path}")
                response_parts.append(f"\nType: {doc_type}")
                response_parts.append(f"\nSize: {doc_info.get('size', 0) / 1024:.1f} KB")

                if doc_type == "text":
                    response_parts.append("\nContent:")
                    response_parts.append("-" * 80)
                    response_parts.append(f"\n{doc_info.get('content', '')}")

                elif doc_type == "image":
                    if doc_info.get("size_exceeded"):
                        response_parts.append("\n(Size exceeds limit)")
                        response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")
                    elif include_base64:
                        response_parts.append(
                            f"\nBase64 Content: {doc_info.get('content', '')}"
                        )
                        if "url" in doc_info:
                            response_parts.append(f"\nURL: {doc_info['url']}")
                    else:
                        response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")

                response_parts.append(f"\n{'=' * 80}")

        return [TextContent(type="text", text="\n".join(response_parts))]

    async def _handle_list_skills(
        self, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle list_skills tool calls."""
        response_parts: list[str] = []

        status_msg = self.loading_state.get_status_message()
        if status_msg:
            response_parts.append(status_msg)

        if not self.search_engine.skills:
            if not self.loading_state.is_complete:
                return [
                    TextContent(
                        type="text",
                        text=(status_msg or "")
                        + "No skills loaded yet. Please wait for skills to load.",
                    )
                ]
            return [TextContent(type="text", text="No skills currently loaded.")]

        response_parts.extend(
            [
                f"Total skills loaded: {len(self.search_engine.skills)}\n",
                "=" * 80,
                "\n",
            ]
        )

        for i, skill in enumerate(self.search_engine.skills, 1):
            source = skill.source
            if "github.com" in source:
                match = re.search(r"github\.com/([^/]+/[^/]+)", source)
                if match:
                    source = match.group(1)

            doc_count = len(skill.documents)

            response_parts.append(f"{i}. {skill.name}")
            response_parts.append(f"   Description: {skill.description}")
            response_parts.append(f"   Source: {source}")
            response_parts.append(f"   Documents: {doc_count} file(s)")
            response_parts.append("")

        return [TextContent(type="text", text="\n".join(response_parts))]

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        logger.info("Starting MCP server with stdio transport")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


# === Standalone Handler Functions for HTTP Server ===


async def handle_search_skills(
    arguments: dict[str, Any],
    search_engine: SkillSearchEngine,
    loading_state: LoadingState | None,
    default_top_k: int = 3,
    max_content_chars: int | None = None,
) -> list[TextContent]:
    """Handle find_helpful_skills tool calls (standalone version for HTTP server)."""
    task_description = arguments.get("task_description")
    if not task_description:
        raise ValueError("task_description is required")

    top_k = arguments.get("top_k", default_top_k)
    list_documents = arguments.get("list_documents", True)

    response_parts: list[str] = []

    status_msg = loading_state.get_status_message() if loading_state else None
    if status_msg:
        response_parts.append(status_msg)

    results = search_engine.search_dict(task_description, top_k)

    if not results:
        if (
            loading_state
            and not loading_state.is_complete
            and loading_state.loaded_skills == 0
        ):
            return [
                TextContent(
                    type="text",
                    text=(status_msg or "")
                    + "No skills loaded yet. Please wait for skills to load and try again.",
                )
            ]
        return [
            TextContent(
                type="text",
                text="No relevant skills found for the given task description.",
            )
        ]

    response_parts.append(
        f"Found {len(results)} relevant skill(s) for: '{task_description}'\n"
    )

    for i, result in enumerate(results, 1):
        response_parts.append(f"\n{'=' * 80}")
        response_parts.append(f"\nSkill {i}: {result['name']}")
        response_parts.append(f"\nRelevance Score: {result['relevance_score']:.4f}")
        response_parts.append(f"\nSource: {result['source']}")
        response_parts.append(f"\nDescription: {result['description']}")

        documents = result.get("documents", {})
        if documents:
            response_parts.append(f"\nAdditional Documents: {len(documents)} file(s)")

            if list_documents:
                response_parts.append("\nAvailable Documents:")
                for doc_path in sorted(documents.keys()):
                    doc_info = documents[doc_path]
                    doc_type = doc_info.get("type", "unknown")
                    doc_size = doc_info.get("size", 0)
                    size_kb = doc_size / 1024
                    response_parts.append(
                        f"  - {doc_path} ({doc_type}, {size_kb:.1f} KB)"
                    )

        response_parts.append(f"\n{'-' * 80}")
        response_parts.append("\nFull Content:\n")

        content = result["content"]
        if max_content_chars is not None and len(content) > max_content_chars:
            truncated_content = content[:max_content_chars] + "..."
            response_parts.append(truncated_content)
            response_parts.append(
                f"\n\n[Content truncated at {max_content_chars} characters. "
                f"View full skill at: {result['source']}]"
            )
        else:
            response_parts.append(content)

        response_parts.append(f"\n{'=' * 80}\n")

    return [TextContent(type="text", text="\n".join(response_parts))]


async def handle_read_skill_document(
    arguments: dict[str, Any], search_engine: SkillSearchEngine
) -> list[TextContent]:
    """Handle read_skill_document tool calls (standalone version for HTTP server)."""
    skill_name = arguments.get("skill_name")
    if not skill_name:
        raise ValueError("skill_name is required")

    document_path = arguments.get("document_path")
    include_base64 = arguments.get("include_base64", False)

    skill = None
    for s in search_engine.skills:
        if s.name == skill_name:
            skill = s
            break

    if not skill:
        return [
            TextContent(
                type="text",
                text=(
                    f"Skill '{skill_name}' not found. Please use find_helpful_skills "
                    "to find valid skill names."
                ),
            )
        ]

    if not document_path:
        if not skill.documents:
            return [
                TextContent(
                    type="text",
                    text=f"Skill '{skill_name}' has no additional documents.",
                )
            ]

        response_parts = [f"Available documents for skill '{skill_name}':\n"]
        for doc_path, doc_info in sorted(skill.documents.items()):
            doc_type = doc_info.get("type", "unknown")
            doc_size = doc_info.get("size", 0)
            size_kb = doc_size / 1024
            response_parts.append(f"  - {doc_path} ({doc_type}, {size_kb:.1f} KB)")

        return [TextContent(type="text", text="\n".join(response_parts))]

    matching_docs: dict[str, dict[str, Any]] = {}
    for doc_path, doc_info in skill.documents.items():
        if fnmatch.fnmatch(doc_path, document_path) or doc_path == document_path:
            matching_docs[doc_path] = doc_info

    if not matching_docs:
        return [
            TextContent(
                type="text",
                text=(
                    f"No documents matching '{document_path}' found in "
                    f"skill '{skill_name}'."
                ),
            )
        ]

    for doc_path in matching_docs:
        doc_info = matching_docs[doc_path]
        if not doc_info.get("fetched") and "content" not in doc_info:
            content = skill.get_document(doc_path)
            if content:
                matching_docs[doc_path] = content

    response_parts: list[str] = []

    if len(matching_docs) == 1:
        doc_path, doc_info = list(matching_docs.items())[0]
        doc_type = doc_info.get("type")

        if doc_type == "text":
            response_parts.append(f"Document: {doc_path}\n")
            response_parts.append("=" * 80)
            response_parts.append(f"\n{doc_info.get('content', '')}")

        elif doc_type == "image":
            response_parts.append(f"Image: {doc_path}\n")
            if doc_info.get("size_exceeded"):
                response_parts.append(
                    f"Size: {doc_info.get('size', 0) / 1024:.1f} KB (exceeds limit)"
                )
                response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")
            elif include_base64:
                response_parts.append(
                    f"Base64 Content:\n{doc_info.get('content', '')}"
                )
                if "url" in doc_info:
                    response_parts.append(
                        f"\n\nAlternatively, access via URL: {doc_info['url']}"
                    )
            else:
                response_parts.append(f"URL: {doc_info.get('url', 'N/A')}")
                if "content" in doc_info:
                    response_parts.append(
                        "\n(Set include_base64=true to get base64-encoded content)"
                    )

    else:
        response_parts.append(
            f"Found {len(matching_docs)} documents matching '{document_path}':\n"
        )

        for doc_path, doc_info in sorted(matching_docs.items()):
            doc_type = doc_info.get("type")
            response_parts.append(f"\n{'=' * 80}")
            response_parts.append(f"\nDocument: {doc_path}")
            response_parts.append(f"\nType: {doc_type}")
            response_parts.append(f"\nSize: {doc_info.get('size', 0) / 1024:.1f} KB")

            if doc_type == "text":
                response_parts.append("\nContent:")
                response_parts.append("-" * 80)
                response_parts.append(f"\n{doc_info.get('content', '')}")

            elif doc_type == "image":
                if doc_info.get("size_exceeded"):
                    response_parts.append("\n(Size exceeds limit)")
                    response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")
                elif include_base64:
                    response_parts.append(
                        f"\nBase64 Content: {doc_info.get('content', '')}"
                    )
                    if "url" in doc_info:
                        response_parts.append(f"\nURL: {doc_info['url']}")
                else:
                    response_parts.append(f"\nURL: {doc_info.get('url', 'N/A')}")

            response_parts.append(f"\n{'=' * 80}")

    return [TextContent(type="text", text="\n".join(response_parts))]


async def handle_list_skills(
    arguments: dict[str, Any],
    search_engine: SkillSearchEngine,
    loading_state: LoadingState | None,
) -> list[TextContent]:
    """Handle list_skills tool calls (standalone version for HTTP server)."""
    response_parts: list[str] = []

    status_msg = loading_state.get_status_message() if loading_state else None
    if status_msg:
        response_parts.append(status_msg)

    if not search_engine.skills:
        if loading_state and not loading_state.is_complete:
            return [
                TextContent(
                    type="text",
                    text=(status_msg or "")
                    + "No skills loaded yet. Please wait for skills to load.",
                )
            ]
        return [TextContent(type="text", text="No skills currently loaded.")]

    response_parts.extend(
        [
            f"Total skills loaded: {len(search_engine.skills)}\n",
            "=" * 80,
            "\n",
        ]
    )

    for i, skill in enumerate(search_engine.skills, 1):
        source = skill.source
        if "github.com" in source:
            match = re.search(r"github\.com/([^/]+/[^/]+)", source)
            if match:
                source = match.group(1)

        doc_count = len(skill.documents)

        response_parts.append(f"{i}. {skill.name}")
        response_parts.append(f"   Description: {skill.description}")
        response_parts.append(f"   Source: {source}")
        response_parts.append(f"   Documents: {doc_count} file(s)")
        response_parts.append("")

    return [TextContent(type="text", text="\n".join(response_parts))]

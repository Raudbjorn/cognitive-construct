"""HTTP server with MCP Streamable HTTP transport using FastMCP."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import config_to_dict, load_config
from .scheduler import HourlyScheduler
from .search_engine import SkillSearchEngine
from .skill_loader import load_all_skills, load_skills_in_batches
from .types import Config, LoadingState
from .update_checker import UpdateChecker

logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("skill-search-backend")

# Global state
search_engine: SkillSearchEngine | None = None
loading_state_global: LoadingState | None = None
update_checker_global: UpdateChecker | None = None
scheduler_global: HourlyScheduler | None = None
config_global: Config | None = None


def register_mcp_tools(
    default_top_k: int = 3, max_content_chars: int | None = None
) -> None:
    """Register MCP tools using FastMCP decorators."""
    from .mcp_handlers import (
        handle_list_skills,
        handle_read_skill_document,
        handle_search_skills,
    )

    @mcp.tool(
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
    )
    async def find_helpful_skills(
        task_description: str,
        top_k: int = default_top_k,
        list_documents: bool = True,
    ) -> list[TextContent]:
        """Search for relevant skills."""
        return await handle_search_skills(
            {
                "task_description": task_description,
                "top_k": top_k,
                "list_documents": list_documents,
            },
            search_engine,
            loading_state_global,
            default_top_k,
            max_content_chars,
        )

    @mcp.tool(
        name="read_skill_document",
        title="Open skill documents and assets",
        description=(
            "Use after finding a relevant skill to retrieve specific documents "
            "(scripts, references, assets). Supports pattern matching "
            "(e.g., 'scripts/*.py') to fetch multiple files. Returns text content "
            "or URLs and never executes code. Prefer pulling only the files you "
            "need to complete the current step."
        ),
    )
    async def read_skill_document(
        skill_name: str,
        document_path: str | None = None,
        include_base64: bool = False,
    ) -> list[TextContent]:
        """Read a document from a skill."""
        args: dict[str, Any] = {
            "skill_name": skill_name,
            "include_base64": include_base64,
        }
        if document_path is not None:
            args["document_path"] = document_path
        return await handle_read_skill_document(args, search_engine)

    @mcp.tool(
        name="list_skills",
        title="List available skills",
        description=(
            "Returns the full inventory of loaded skills (names, descriptions, "
            "sources, document counts) for exploration or debugging. For task-driven "
            "work, prefer calling 'find_helpful_skills' first to locate the most "
            "relevant option before reading documents."
        ),
    )
    async def list_skills() -> list[TextContent]:
        """List all loaded skills."""
        return await handle_list_skills({}, search_engine, loading_state_global)


async def health_check(request: Any) -> JSONResponse:
    """Health check endpoint."""
    skills_loaded = len(search_engine.skills) if search_engine else 0
    models_loaded = search_engine.model is not None if search_engine else False

    response: dict[str, Any] = {
        "status": "ok",
        "version": "1.0.6",
        "skills_loaded": skills_loaded,
        "models_loaded": models_loaded,
        "loading_complete": (
            loading_state_global.is_complete if loading_state_global else False
        ),
    }

    # Add auto-update information
    if config_global:
        response["auto_update_enabled"] = config_global.auto_update_enabled

    if scheduler_global:
        scheduler_status = scheduler_global.get_status()
        response.update(
            {
                "next_update_check": scheduler_status.get("next_run_time"),
                "last_update_check": scheduler_status.get("last_run_time"),
            }
        )

    if update_checker_global:
        api_usage = update_checker_global.get_api_usage()
        response.update(
            {
                "github_api_calls_this_hour": api_usage.get("calls_this_hour", 0),
                "github_api_limit": api_usage.get("limit_per_hour", 60),
                "github_authenticated": api_usage.get("authenticated", False),
            }
        )

    if loading_state_global:
        with loading_state_global._lock:
            if loading_state_global.errors:
                response["update_errors"] = loading_state_global.errors[-5:]

    return JSONResponse(response)


async def initialize_backend(
    config_path: str | None = None, verbose: bool = False
) -> None:
    """Initialize search engine and load skills."""
    global search_engine
    global loading_state_global
    global update_checker_global
    global scheduler_global
    global config_global

    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Initializing Claude Skills MCP Backend")

    # Load configuration
    config_result = load_config(config_path)
    if config_result.is_err():
        logger.error(f"Failed to load config: {config_result.error.message}")
        raise RuntimeError(f"Configuration error: {config_result.error.message}")

    config = config_result.value
    config_global = config
    config_dict = config_to_dict(config)

    # Initialize search engine
    logger.info("Initializing search engine...")
    search_engine = SkillSearchEngine(config.embedding_model)

    # Initialize loading state
    loading_state_global = LoadingState()

    # Initialize update checker
    github_token = config.github_api_token
    update_checker_global = UpdateChecker(github_token)
    logger.info(
        f"Update checker initialized "
        f"(GitHub token: {'provided' if github_token else 'not provided'})"
    )

    # Register MCP tools
    register_mcp_tools(
        default_top_k=config.default_top_k,
        max_content_chars=config.max_skill_content_chars,
    )

    # Define batch callback for incremental loading
    def on_batch_loaded(batch_skills: list[Any], total_loaded: int) -> None:
        logger.info(
            f"Batch loaded: {len(batch_skills)} skills (total: {total_loaded})"
        )
        search_engine.add_skills(batch_skills)
        loading_state_global.update_progress(total_loaded)

    # Start background thread to load skills
    def background_loader() -> None:
        try:
            logger.info("Starting background skill loading...")
            load_skills_in_batches(
                skill_sources=config_dict["skill_sources"],
                config=config_dict,
                batch_callback=on_batch_loaded,
                batch_size=config_dict.get("batch_size", 10),
            )
            loading_state_global.mark_complete()
            logger.info("Background skill loading complete")
        except Exception as e:
            logger.error(f"Error in background loading: {e}", exc_info=True)
            loading_state_global.add_error(str(e))
            loading_state_global.mark_complete()

    # Start the background loading thread
    loader_thread = threading.Thread(target=background_loader, daemon=True)
    loader_thread.start()
    logger.info("Background loading thread started, server is ready")

    # Setup auto-update scheduler if enabled
    if config.auto_update_enabled:
        interval_minutes = config.auto_update_interval_minutes

        def perform_sync_update() -> None:
            """Synchronous update logic to run in a thread."""
            try:
                logger.info("Running scheduled update check...")

                # Check for updates
                result = update_checker_global.check_for_updates(
                    config_dict["skill_sources"]
                )

                logger.info(
                    f"Update check complete: {len(result.changed_sources)} sources "
                    f"changed, {result.api_calls_made} API calls made"
                )

                if result.errors:
                    for error in result.errors:
                        logger.warning(f"Update check error: {error}")
                        loading_state_global.add_error(error)

                # Reload skills if updates detected
                if result.has_updates:
                    logger.info(
                        f"Reloading {len(result.changed_sources)} changed sources..."
                    )

                    logger.info("Reloading all skills...")
                    new_skills = load_all_skills(
                        skill_sources=config_dict["skill_sources"],
                        config=config_dict,
                    )

                    # Re-index all skills
                    search_engine.index_skills(new_skills)
                    logger.info(f"Re-indexed {len(new_skills)} skills after update")
                else:
                    logger.info("No updates detected")

                # Warn if approaching API limit (only for non-authenticated)
                api_usage = update_checker_global.get_api_usage()
                if (
                    not api_usage["authenticated"]
                    and api_usage["calls_this_hour"] >= 50
                ):
                    logger.warning(
                        f"Approaching GitHub API rate limit: "
                        f"{api_usage['calls_this_hour']}/60 calls this hour"
                    )

            except Exception as e:
                error_msg = f"Error during scheduled update: {e}"
                logger.error(error_msg, exc_info=True)
                loading_state_global.add_error(error_msg)

        async def update_callback() -> None:
            """Callback for scheduled updates."""
            await asyncio.to_thread(perform_sync_update)

        # Create and start scheduler
        scheduler_global = HourlyScheduler(interval_minutes, update_callback)
        scheduler_global.start()
        logger.info(
            f"Auto-update scheduler started (interval: {interval_minutes} minutes)"
        )
    else:
        logger.info("Auto-update disabled in configuration")


async def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: str | None = None,
    verbose: bool = False,
) -> None:
    """Run the HTTP server using FastMCP with custom routes."""
    # Initialize backend (search engine, skills, etc.)
    await initialize_backend(config_path, verbose)

    # Get FastMCP's ASGI app (includes /mcp route internally)
    fastmcp_app = mcp.streamable_http_app()

    # Add our custom health route to the FastMCP app
    fastmcp_app.routes.insert(0, Route("/health", health_check, methods=["GET"]))

    app = fastmcp_app

    # Run server with uvicorn
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="debug" if verbose else "info",
    )
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

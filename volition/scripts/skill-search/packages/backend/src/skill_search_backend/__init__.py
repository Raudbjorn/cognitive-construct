"""Claude Skills MCP Backend - Semantic skill search server."""

from __future__ import annotations

__version__ = "1.0.6"

# Core types
from .result import Err, Ok, Result
from .types import (
    ApiError,
    Config,
    ConfigError,
    DocumentContent,
    DocumentMetadata,
    DocumentType,
    LoadError,
    LoadingState,
    SearchError,
    SearchResponse,
    SearchResult,
    Skill,
    SkillSourceConfig,
    SourceType,
)

# Configuration
from .config import config_to_dict, get_example_config, load_config

# Core components
from .search_engine import SkillSearchEngine
from .skill_loader import (
    load_all_skills,
    load_from_github,
    load_from_local,
    load_skills_in_batches,
    parse_skill_md,
)

# MCP handlers
from .mcp_handlers import (
    SkillsMCPServer,
    handle_list_skills,
    handle_read_skill_document,
    handle_search_skills,
)

# Server
from .http_server import initialize_backend, run_server

# Update system
from .scheduler import HourlyScheduler
from .state_manager import StateManager
from .update_checker import UpdateChecker, UpdateResult

__all__ = [
    # Version
    "__version__",
    # Result types
    "Ok",
    "Err",
    "Result",
    # Data types
    "ApiError",
    "Config",
    "ConfigError",
    "DocumentContent",
    "DocumentMetadata",
    "DocumentType",
    "LoadError",
    "LoadingState",
    "SearchError",
    "SearchResponse",
    "SearchResult",
    "Skill",
    "SkillSourceConfig",
    "SourceType",
    # Configuration
    "config_to_dict",
    "get_example_config",
    "load_config",
    # Core components
    "SkillSearchEngine",
    "load_all_skills",
    "load_from_github",
    "load_from_local",
    "load_skills_in_batches",
    "parse_skill_md",
    # MCP handlers
    "SkillsMCPServer",
    "handle_list_skills",
    "handle_read_skill_document",
    "handle_search_skills",
    # Server
    "initialize_backend",
    "run_server",
    # Update system
    "HourlyScheduler",
    "StateManager",
    "UpdateChecker",
    "UpdateResult",
]

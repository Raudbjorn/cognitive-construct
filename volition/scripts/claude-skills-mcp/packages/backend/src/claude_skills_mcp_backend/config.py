"""Configuration management for Claude Skills MCP server."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .result import Err, Ok, Result
from .types import Config, ConfigError, SkillSourceConfig, SourceType

logger = logging.getLogger(__name__)

# === Constants ===

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
DEFAULT_MAX_IMAGE_SIZE_BYTES = 5242880  # 5MB
DEFAULT_UPDATE_INTERVAL_MINUTES = 60

DEFAULT_SKILL_SOURCES: tuple[SkillSourceConfig, ...] = (
    SkillSourceConfig(
        type=SourceType.GITHUB,
        url="https://github.com/anthropics/skills",
    ),
    SkillSourceConfig(
        type=SourceType.GITHUB,
        url="https://github.com/K-Dense-AI/claude-scientific-skills",
    ),
    SkillSourceConfig(
        type=SourceType.LOCAL,
        path="~/.claude/skills",
    ),
)

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
)

DEFAULT_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".py",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".r",
    ".ipynb",
    ".xml",
)


def _get_default_config() -> Config:
    """Get the default configuration.

    Returns
    -------
    Config
        Default configuration object.
    """
    return Config(
        skill_sources=DEFAULT_SKILL_SOURCES,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_top_k=DEFAULT_TOP_K,
        max_skill_content_chars=None,
        load_skill_documents=True,
        max_image_size_bytes=DEFAULT_MAX_IMAGE_SIZE_BYTES,
        allowed_image_extensions=DEFAULT_IMAGE_EXTENSIONS,
        text_file_extensions=DEFAULT_TEXT_EXTENSIONS,
        auto_update_enabled=True,
        auto_update_interval_minutes=DEFAULT_UPDATE_INTERVAL_MINUTES,
        github_api_token=None,
    )


def _parse_skill_source(source_dict: dict[str, Any]) -> SkillSourceConfig | None:
    """Parse a skill source configuration dictionary.

    Parameters
    ----------
    source_dict : dict[str, Any]
        Dictionary with source configuration.

    Returns
    -------
    SkillSourceConfig | None
        Parsed source config, or None if invalid.
    """
    source_type_str = source_dict.get("type")
    if not source_type_str:
        return None

    try:
        source_type = SourceType(source_type_str)
    except ValueError:
        logger.warning(f"Unknown source type: {source_type_str}")
        return None

    return SkillSourceConfig(
        type=source_type,
        url=source_dict.get("url"),
        path=source_dict.get("path"),
        subpath=source_dict.get("subpath", ""),
    )


def _parse_config_dict(config_dict: dict[str, Any]) -> Config:
    """Parse a configuration dictionary into a Config object.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Raw configuration dictionary.

    Returns
    -------
    Config
        Parsed configuration object.
    """
    defaults = _get_default_config()

    # Parse skill sources
    skill_sources_raw = config_dict.get("skill_sources", [])
    skill_sources: list[SkillSourceConfig] = []
    for source_dict in skill_sources_raw:
        source = _parse_skill_source(source_dict)
        if source:
            skill_sources.append(source)

    # Use defaults if no valid sources
    if not skill_sources:
        skill_sources = list(defaults.skill_sources)

    # Parse image extensions
    image_ext_raw = config_dict.get("allowed_image_extensions")
    image_extensions = (
        tuple(image_ext_raw)
        if image_ext_raw
        else defaults.allowed_image_extensions
    )

    # Parse text extensions
    text_ext_raw = config_dict.get("text_file_extensions")
    text_extensions = (
        tuple(text_ext_raw)
        if text_ext_raw
        else defaults.text_file_extensions
    )

    return Config(
        skill_sources=tuple(skill_sources),
        embedding_model=config_dict.get("embedding_model", defaults.embedding_model),
        default_top_k=config_dict.get("default_top_k", defaults.default_top_k),
        max_skill_content_chars=config_dict.get("max_skill_content_chars"),
        load_skill_documents=config_dict.get(
            "load_skill_documents", defaults.load_skill_documents
        ),
        max_image_size_bytes=config_dict.get(
            "max_image_size_bytes", defaults.max_image_size_bytes
        ),
        allowed_image_extensions=image_extensions,
        text_file_extensions=text_extensions,
        auto_update_enabled=config_dict.get(
            "auto_update_enabled", defaults.auto_update_enabled
        ),
        auto_update_interval_minutes=config_dict.get(
            "auto_update_interval_minutes", defaults.auto_update_interval_minutes
        ),
        github_api_token=config_dict.get("github_api_token"),
    )


def load_config(config_path: str | None = None) -> Result[Config, ConfigError]:
    """Load configuration from file or use defaults.

    Parameters
    ----------
    config_path : str | None
        Path to configuration JSON file. If None, uses default config.

    Returns
    -------
    Result[Config, ConfigError]
        Ok with Config on success, Err with ConfigError on failure.
    """
    if config_path is None:
        logger.info("No config file specified, using default configuration")
        return Ok(_get_default_config())

    config_file = Path(config_path).expanduser().resolve()

    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Ok(_get_default_config())

    try:
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        config = _parse_config_dict(config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return Ok(config)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in config file: {e}"
        logger.error(error_msg)
        return Err(ConfigError(message=error_msg, path=config_path))

    except Exception as e:
        error_msg = f"Error loading config: {e}"
        logger.error(error_msg)
        return Err(ConfigError(message=error_msg, path=config_path))


def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert Config object to dictionary for compatibility.

    This is useful for interfacing with code that expects the old dict format.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    dict[str, Any]
        Configuration as dictionary.
    """
    return {
        "skill_sources": [
            {
                "type": source.type.value,
                "url": source.url,
                "path": source.path,
                "subpath": source.subpath,
            }
            for source in config.skill_sources
        ],
        "embedding_model": config.embedding_model,
        "default_top_k": config.default_top_k,
        "max_skill_content_chars": config.max_skill_content_chars,
        "load_skill_documents": config.load_skill_documents,
        "max_image_size_bytes": config.max_image_size_bytes,
        "allowed_image_extensions": list(config.allowed_image_extensions),
        "text_file_extensions": list(config.text_file_extensions),
        "auto_update_enabled": config.auto_update_enabled,
        "auto_update_interval_minutes": config.auto_update_interval_minutes,
        "github_api_token": config.github_api_token,
    }


def get_example_config() -> str:
    """Get the default configuration as JSON string.

    Returns
    -------
    str
        Default configuration in JSON format.
    """
    config_with_comments = {
        "skill_sources": [
            {
                "type": "github",
                "url": "https://github.com/anthropics/skills",
                "comment": "Official Anthropic skills - diverse examples",
            },
            {
                "type": "github",
                "url": "https://github.com/K-Dense-AI/claude-scientific-skills",
                "comment": "70+ scientific skills for bioinformatics and analysis",
            },
            {
                "type": "local",
                "path": "~/.claude/skills",
                "comment": "Your custom local skills (optional)",
            },
        ],
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "default_top_k": DEFAULT_TOP_K,
        "max_skill_content_chars": None,
        "comment_max_chars": "Set to integer to truncate, null for unlimited",
        "load_skill_documents": True,
        "max_image_size_bytes": DEFAULT_MAX_IMAGE_SIZE_BYTES,
        "allowed_image_extensions": list(DEFAULT_IMAGE_EXTENSIONS),
        "text_file_extensions": list(DEFAULT_TEXT_EXTENSIONS),
        "auto_update_enabled": True,
        "comment_auto_update": "Enable automatic hourly skill updates",
        "auto_update_interval_minutes": DEFAULT_UPDATE_INTERVAL_MINUTES,
        "comment_interval": "Check for updates every N minutes",
        "github_api_token": None,
        "comment_token": "Optional GitHub token for 5000 req/hr (default: 60 req/hr)",
    }
    return json.dumps(config_with_comments, indent=2)

"""Simplified config matching CGC's config_manager API.

Reads env vars with CGC_ prefix, then falls back to defaults.
No file-based config, no rich UI — just what graph_builder needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Matches CGC DEFAULT_CONFIG keys used by graph_builder / language parsers
DEFAULT_CONFIG: dict[str, str] = {
    "INDEX_VARIABLES": "true",
    "INDEX_SOURCE": "true",
    "MAX_FILE_SIZE_MB": "10",
    "IGNORE_TEST_FILES": "false",
    "IGNORE_HIDDEN_FILES": "true",
    "IGNORE_DIRS": (
        "node_modules,venv,.venv,env,.env,dist,build,target,out,"
        ".git,.idea,.vscode,__pycache__"
    ),
    "MAX_DEPTH": "unlimited",
    "PARALLEL_WORKERS": "4",
    "COMPLEXITY_THRESHOLD": "10",
    "DEBUG_LOGS": "false",
    "ENABLE_APP_LOGS": "CRITICAL",
}


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Runtime indexing configuration."""

    index_variables: bool = True
    index_source: bool = True
    max_file_size_mb: int = 10
    ignore_test_files: bool = False
    ignore_hidden_files: bool = True
    ignore_dirs: str = (
        "node_modules,venv,.venv,env,.env,dist,build,target,out,"
        ".git,.idea,.vscode,__pycache__"
    )
    max_depth: str = "unlimited"
    parallel_workers: int = 4
    complexity_threshold: int = 10


def get_config_value(key: str) -> str | None:
    """Get config value — env var (CGC_<KEY>) first, then defaults."""
    env_val = os.environ.get(key) or os.environ.get(f"CGC_{key}")
    if env_val is not None:
        return env_val
    return DEFAULT_CONFIG.get(key)

"""Configuration for inmemoria library."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class Config:
    """Configuration for inmemoria.

    All settings can be overridden via environment variables prefixed with INMEMORIA_.
    """
    # Database settings
    db_filename: str = "inmemoria.db"
    db_pool_size: int = 5
    db_busy_timeout_ms: int = 5000

    # Performance settings
    batch_size: int = 100
    max_concurrency: int = 4
    analysis_timeout_seconds: float = 300.0

    # Analysis settings
    max_file_size_bytes: int = 1_000_000  # 1MB
    skip_patterns: list[str] = field(default_factory=lambda: [
        "node_modules",
        ".git",
        "dist",
        "build",
        ".next",
        "__pycache__",
        "venv",
        ".venv",
        "target",
    ])

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes

    # Supported file extensions
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".py",
        ".rs",
        ".go",
        ".java",
        ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
        ".cs",
        ".svelte",
        ".sql",
        ".php", ".phtml",
    }

    # Extension to language mapping
    EXTENSION_LANGUAGE_MAP: ClassVar[dict[str, str]] = {
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cts": "typescript",
        ".mts": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".py": "python",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".svelte": "svelte",
        ".sql": "sql",
        ".php": "php",
        ".phtml": "php",
    }

    @classmethod
    def from_env(cls) -> Config:
        """Create config from environment variables."""
        return cls(
            db_filename=os.environ.get("INMEMORIA_DB_FILENAME", cls.db_filename),
            db_pool_size=int(os.environ.get("INMEMORIA_DB_POOL_SIZE", cls.db_pool_size)),
            db_busy_timeout_ms=int(os.environ.get("INMEMORIA_DB_BUSY_TIMEOUT_MS", cls.db_busy_timeout_ms)),
            batch_size=int(os.environ.get("INMEMORIA_BATCH_SIZE", cls.batch_size)),
            max_concurrency=int(os.environ.get("INMEMORIA_MAX_CONCURRENCY", cls.max_concurrency)),
            analysis_timeout_seconds=float(os.environ.get("INMEMORIA_ANALYSIS_TIMEOUT", cls.analysis_timeout_seconds)),
            max_file_size_bytes=int(os.environ.get("INMEMORIA_MAX_FILE_SIZE", cls.max_file_size_bytes)),
            cache_ttl_seconds=int(os.environ.get("INMEMORIA_CACHE_TTL", cls.cache_ttl_seconds)),
        )

    def get_database_path(self, project_path: str) -> Path:
        """Get database path for a project."""
        project_dir = Path(project_path).resolve()
        inmemoria_dir = project_dir / ".inmemoria"
        inmemoria_dir.mkdir(parents=True, exist_ok=True)
        return inmemoria_dir / self.db_filename

    def detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_LANGUAGE_MAP.get(ext, "unknown")

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file extension is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def should_skip_path(self, path: str) -> bool:
        """Check if path should be skipped."""
        path_parts = Path(path).parts
        return any(skip in path_parts for skip in self.skip_patterns)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config

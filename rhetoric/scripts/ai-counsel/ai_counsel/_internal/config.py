"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class HTTPAdapterConfig(BaseModel):
    """Configuration for HTTP-based adapter."""

    type: str = "http"
    base_url: str
    api_key: str | None = None
    headers: dict[str, str] | None = None
    timeout: int = 60
    max_retries: int = 3

    @field_validator("api_key", "base_url")
    @classmethod
    def resolve_env_vars(cls, v: str | None, info: Any) -> str | None:
        """Resolve ${ENV_VAR} references in string fields.

        For optional fields like api_key:
        - If env var is missing, returns None (allows graceful degradation)

        For required fields like base_url:
        - If env var is missing, raises ValueError
        """
        if v is None:
            return v

        pattern = r"\$\{([^}]+)\}"
        is_api_key = info.field_name == "api_key"

        def replacer(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                if is_api_key:
                    return "__MISSING_API_KEY__"
                raise ValueError(
                    f"Environment variable '{env_var}' is not set. "
                    f"Required for configuration."
                )
            return value

        result = re.sub(pattern, replacer, v)
        if is_api_key and "__MISSING_API_KEY__" in result:
            return None
        return result


class DefaultsConfig(BaseModel):
    """Default settings."""

    mode: str = "quick"
    rounds: int = 2
    max_rounds: int = 5
    timeout_per_round: int = 300


class StorageConfig(BaseModel):
    """Storage configuration."""

    transcripts_dir: str = "transcripts"
    format: str = "markdown"
    auto_export: bool = True


class ConvergenceDetectionConfig(BaseModel):
    """Convergence detection configuration."""

    enabled: bool = True
    semantic_similarity_threshold: float = 0.85
    divergence_threshold: float = 0.40
    min_rounds_before_check: int = 1
    consecutive_stable_rounds: int = 2
    stance_stability_threshold: float = 0.90
    response_length_drop_threshold: float = 0.50


class EarlyStoppingConfig(BaseModel):
    """Model-controlled early stopping configuration."""

    enabled: bool = True
    threshold: float = 0.66
    respect_min_rounds: bool = True


class FileTreeConfig(BaseModel):
    """Configuration for file tree generation in Round 1 prompts."""

    max_depth: int = Field(default=3, ge=1, le=10)
    max_files: int = Field(default=100, ge=10, le=1000)
    enabled: bool = True


class ToolSecurityConfig(BaseModel):
    """Security configuration for evidence-based deliberation tools."""

    exclude_patterns: list[str] = Field(
        default=[
            "transcripts/",
            "transcripts/**",
            ".git/",
            ".git/**",
            "node_modules/",
            "node_modules/**",
            ".venv/",
            "venv/",
            "__pycache__/",
        ],
    )
    max_file_size_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)


class DeliberationConfig(BaseModel):
    """Deliberation engine configuration."""

    convergence_detection: ConvergenceDetectionConfig = Field(
        default_factory=ConvergenceDetectionConfig
    )
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    convergence_threshold: float = 0.85
    enable_convergence_detection: bool = True
    tool_context_max_rounds: int = Field(default=2, ge=1, le=10)
    tool_output_max_chars: int = Field(default=1000, ge=100, le=10000)
    file_tree: FileTreeConfig = Field(default_factory=FileTreeConfig)
    tool_security: ToolSecurityConfig = Field(default_factory=ToolSecurityConfig)


class DecisionGraphConfig(BaseModel):
    """Configuration for decision graph memory."""

    enabled: bool = False
    db_path: str = "decision_graph.db"
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_context_decisions: int = Field(default=3, ge=1, le=10)
    compute_similarities: bool = True
    context_token_budget: int = Field(default=1500, ge=500, le=10000)
    tier_boundaries: dict[str, float] = Field(
        default_factory=lambda: {"strong": 0.75, "moderate": 0.60}
    )
    query_window: int = Field(default=1000, ge=50, le=10000)
    query_cache_size: int = Field(default=200, ge=10, le=10000)
    embedding_cache_size: int = Field(default=500, ge=10, le=10000)
    query_ttl: int = Field(default=300, ge=60, le=3600)
    adaptive_k_small_threshold: int = Field(default=100, ge=10, le=1000)
    adaptive_k_medium_threshold: int = Field(default=1000, ge=100, le=10000)
    adaptive_k_small: int = Field(default=5, ge=1, le=20)
    adaptive_k_medium: int = Field(default=3, ge=1, le=20)
    adaptive_k_large: int = Field(default=2, ge=1, le=20)
    noise_floor: float = Field(default=0.40, ge=0.0, le=1.0)

    @field_validator("tier_boundaries")
    @classmethod
    def validate_tier_boundaries(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate tier boundaries: strong > moderate > 0."""
        if not isinstance(v, dict) or "strong" not in v or "moderate" not in v:
            raise ValueError("tier_boundaries must have 'strong' and 'moderate' keys")

        if not (0.0 < v["moderate"] < v["strong"] <= 1.0):
            raise ValueError(
                f"tier_boundaries must satisfy: 0 < moderate ({v['moderate']}) "
                f"< strong ({v['strong']}) <= 1"
            )

        return v

    @field_validator("db_path")
    @classmethod
    def resolve_db_path(cls, v: str) -> str:
        """Resolve db_path to absolute path."""
        pattern = r"\$\{([^}]+)\}"

        def replacer(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(
                    f"Environment variable '{env_var}' is not set. "
                    f"Required for db_path configuration."
                )
            return value

        resolved = re.sub(pattern, replacer, v)
        path = Path(resolved)

        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        return str(path)


class ClientConfig(BaseModel):
    """Client library configuration."""

    adapters: dict[str, HTTPAdapterConfig] = Field(default_factory=dict)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    deliberation: DeliberationConfig = Field(default_factory=DeliberationConfig)
    decision_graph: DecisionGraphConfig | None = None


def load_config(path: str | Path = "config.yaml") -> ClientConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to config file (default: config.yaml)

    Returns:
        Validated ClientConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Extract relevant sections for client config
    client_data: dict[str, Any] = {}

    # Map adapters - only HTTP adapters
    if "adapters" in data:
        http_adapters = {}
        for name, adapter_config in data["adapters"].items():
            if adapter_config.get("type") == "http":
                http_adapters[name] = adapter_config
        if http_adapters:
            client_data["adapters"] = http_adapters

    # Map other sections
    if "defaults" in data:
        client_data["defaults"] = data["defaults"]
    if "storage" in data:
        client_data["storage"] = data["storage"]
    if "deliberation" in data:
        client_data["deliberation"] = data["deliberation"]
    if "decision_graph" in data:
        client_data["decision_graph"] = data["decision_graph"]

    return ClientConfig(**client_data)


def create_default_config(
    ollama_url: str | None = None,
    lmstudio_url: str | None = None,
    openrouter_api_key: str | None = None,
    openrouter_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 300,
) -> ClientConfig:
    """Create a default configuration from parameters.

    Args:
        ollama_url: Ollama API URL (or use OLLAMA_URL env var)
        lmstudio_url: LM Studio API URL (or use LMSTUDIO_URL env var)
        openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
        openrouter_url: OpenRouter API URL
        timeout: Default timeout for adapter calls

    Returns:
        ClientConfig with adapters configured
    """
    adapters: dict[str, HTTPAdapterConfig] = {}

    # Resolve from env vars if not provided
    ollama_url = ollama_url or os.getenv("OLLAMA_URL")
    lmstudio_url = lmstudio_url or os.getenv("LMSTUDIO_URL")
    openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

    if ollama_url:
        adapters["ollama"] = HTTPAdapterConfig(
            base_url=ollama_url,
            timeout=timeout,
        )

    if lmstudio_url:
        adapters["lmstudio"] = HTTPAdapterConfig(
            base_url=lmstudio_url,
            timeout=timeout,
        )

    if openrouter_api_key:
        adapters["openrouter"] = HTTPAdapterConfig(
            base_url=openrouter_url,
            api_key=openrouter_api_key,
            timeout=timeout,
        )

    return ClientConfig(adapters=adapters)

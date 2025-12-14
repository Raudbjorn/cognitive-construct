"""Configuration for Exa API."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class APIConfig:
    """API configuration constants."""
    BASE_URL: str = "https://api.exa.ai"
    SEARCH_ENDPOINT: str = "/search"
    CONTEXT_ENDPOINT: str = "/context"
    RESEARCH_ENDPOINT: str = "/research/v0/tasks"
    DEFAULT_NUM_RESULTS: int = 8
    DEFAULT_MAX_CHARACTERS: int = 10000
    DEFAULT_TOKENS: int = 5000
    WEB_SEARCH_TIMEOUT_SECONDS: float = 25.0
    CODE_SEARCH_TIMEOUT_SECONDS: float = 30.0
    RESEARCH_TIMEOUT_SECONDS: float = 300.0


API_CONFIG = APIConfig()

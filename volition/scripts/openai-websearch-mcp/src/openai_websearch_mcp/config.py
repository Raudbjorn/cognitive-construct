"""Configuration constants."""

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120.0  # Search might take longer
API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_MODEL_ENV_VAR = "OPENAI_DEFAULT_MODEL"

REASONING_MODELS = frozenset({"gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini"})
VALID_MODELS = frozenset({"gpt-4o", "gpt-4o-mini"}) | REASONING_MODELS

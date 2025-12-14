"""Configuration constants."""

from .types import ProviderConfig

PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        env_key="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-haiku-20240307",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-1.5-flash",
    ),
    "grok": ProviderConfig(
        name="grok",
        env_key="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-beta",
    ),
    "mistral": ProviderConfig(
        name="mistral",
        env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
    ),
}

# Tag-based routing preferences
TAG_ROUTING: dict[str, list[str]] = {
    "coding": ["deepseek", "anthropic", "openai"],
    "reasoning": ["deepseek", "anthropic", "openai"],
    "creative": ["openai", "anthropic", "gemini"],
    "business": ["openai", "anthropic", "mistral"],
    "general": ["openai", "anthropic", "deepseek"],
    "math": ["deepseek", "openai", "anthropic"],
}

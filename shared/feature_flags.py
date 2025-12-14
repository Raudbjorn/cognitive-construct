"""
Feature Flags for Claude Skills.

Captured during Phase 0.4 Infrastructure Decisions (2025-12-13).
See specs/design.md Appendix E for decision rationale.
"""
import os
from typing import Dict, Any


FEATURE_FLAGS = {
    # Infrastructure backends
    "MEM0_HOSTED": True,            # Use hosted mem0 API (not self-hosted)
    "MEM0_SELF_HOSTED": False,      # Self-hosted mem0 with PostgreSQL
    "CODEGRAPH_ENABLED": False,     # Neo4j-based code graph (requires NEO4J_*)
    "SEARXNG_ENABLED": False,       # Self-hosted meta-search (requires SEARXNG_URL)

    # Search providers
    "KAGI_ENABLED": False,          # Kagi search (closed beta)
    "EXA_ENABLED": True,            # Exa neural search
    "PERPLEXITY_ENABLED": True,     # Perplexity conversational search
    "BRAVE_ENABLED": True,          # Brave search
    "SERPER_ENABLED": True,         # Serper/Google search
    "TAVILY_ENABLED": True,         # Tavily search

    # LLM providers
    "XAI_ENABLED": False,           # xAI/Grok (no credits)
    "KIMI_ENABLED": False,          # Kimi/Moonshot (auth failed)

    # Security & optional features
    "SHODAN_ENABLED": False,        # Security scanning (requires consent)
    "LIVEKIT_ENABLED": True,        # Real-time audio
    "ELEVENLABS_ENABLED": True,     # Text-to-speech

    # Synergies (Requirement 8.1-8.5)
    "SYNERGIES_ENABLED": True,      # Enable inter-skill communication
    "RHETORIC_ENCYCLOPEDIA_SYNERGY": True,   # Rhetoric -> Encyclopedia context lookup
    "VOLITION_INLAND_EMPIRE_SYNERGY": True,  # Volition -> Inland Empire action logging
    "ENCYCLOPEDIA_CACHE_SYNERGY": True,      # Encyclopedia -> Inland Empire caching
}


def get_flag(name: str, default: bool = False) -> bool:
    """
    Get a feature flag value.
    Checks environment override first, then falls back to FEATURE_FLAGS.
    """
    # Environment override: NOCP_FLAG_<NAME>=1 or NOCP_FLAG_<NAME>=0
    env_key = f"NOCP_FLAG_{name}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.lower() in ("1", "true", "yes", "on")

    return FEATURE_FLAGS.get(name, default)


def set_flag(name: str, value: bool) -> None:
    """
    Temporarily set a feature flag (in-memory only).
    Useful for testing.
    """
    FEATURE_FLAGS[name] = value


def get_all_flags() -> Dict[str, bool]:
    """Get all feature flags with their current values."""
    return {name: get_flag(name) for name in FEATURE_FLAGS}


def get_enabled_search_providers() -> list[str]:
    """Returns list of enabled search providers for Encyclopedia."""
    providers = []
    if get_flag("EXA_ENABLED"):
        providers.append("exa")
    if get_flag("PERPLEXITY_ENABLED"):
        providers.append("perplexity")
    if get_flag("BRAVE_ENABLED"):
        providers.append("brave")
    if get_flag("SERPER_ENABLED"):
        providers.append("serper")
    if get_flag("TAVILY_ENABLED"):
        providers.append("tavily")
    if get_flag("KAGI_ENABLED"):
        providers.append("kagi")
    if get_flag("SEARXNG_ENABLED"):
        providers.append("searxng")
    return providers


def get_enabled_llm_providers() -> list[str]:
    """Returns list of enabled LLM providers for Rhetoric."""
    # Always-enabled providers (API working per Phase 0.2)
    providers = [
        "anthropic",
        "openai",
        "google",
        "mistral",
        "deepseek",
        "together",
        "openrouter",
        "perplexity",
    ]
    # Conditionally enabled
    if get_flag("XAI_ENABLED"):
        providers.append("xai")
    if get_flag("KIMI_ENABLED"):
        providers.append("kimi")
    return providers


def is_synergy_enabled(synergy_type: str) -> bool:
    """Check if a specific synergy is enabled."""
    if not get_flag("SYNERGIES_ENABLED"):
        return False

    synergy_flags = {
        "rhetoric_encyclopedia": "RHETORIC_ENCYCLOPEDIA_SYNERGY",
        "volition_inland_empire": "VOLITION_INLAND_EMPIRE_SYNERGY",
        "encyclopedia_cache": "ENCYCLOPEDIA_CACHE_SYNERGY",
    }

    flag_name = synergy_flags.get(synergy_type)
    if flag_name:
        return get_flag(flag_name, True)
    return False

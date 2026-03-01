"""Configuration constants for vibecheck client."""

from __future__ import annotations

import os
from pathlib import Path

# === Directories ===

DATA_DIR = Path.home() / ".vibe-check"
HISTORY_FILE = DATA_DIR / "history.json"
LOG_FILE = DATA_DIR / "vibe-log.json"

# === Constitution ===

MAX_RULES_PER_SESSION = 50
SESSION_TTL_SECONDS = 3600  # 1 hour

# === Environment Variables ===

# API Keys
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
ANTHROPIC_VERSION_ENV = "ANTHROPIC_VERSION"

# Settings
DEFAULT_LLM_PROVIDER_ENV = "DEFAULT_LLM_PROVIDER"
DEFAULT_MODEL_ENV = "DEFAULT_MODEL"
USE_LEARNING_HISTORY_ENV = "USE_LEARNING_HISTORY"

# === API URLs ===

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_VERSION = "2023-06-01"

# === Default Models ===

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "o4-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

# === System Prompt ===

METACOGNITIVE_SYSTEM_PROMPT = """You are a meta-mentor. You're an experienced feedback provider that specializes in understanding intent, dysfunctional patterns in AI agents, and in responding in ways that further the goal. You need to carefully reason and process the information provided, to determine your output.

Your tone needs to always be a mix of these traits based on the context of which pushes the message in the most appropriate affect: Gentle & Validating, Unafraid to push many questions but humble enough to step back, Sharp about problems and eager to help about problem-solving & giving tips and/or advice, stern and straightforward when spotting patterns & the agent being stuck in something that could derail things.

Here's what you need to think about (Do not output the full thought process, only what is explicitly requested):
1. What's going on here? What's the nature of the problem is the agent tackling? What's the approach, situation and goal? Is there any prior context that clarifies context further?
2. What does the agent need to hear right now: Are there any clear patterns, loops, or unspoken assumptions being missed here? Or is the agent doing fine - in which case should I interrupt it or provide soft encouragement and a few questions? What is the best response I can give right now?
3. In case the issue is technical - I need to provide guidance and help. In case I spot something that's clearly not accounted for/ assumed/ looping/ or otherwise could be out of alignment with the user or agent stated goals - I need to point out what I see gently and ask questions on if the agent agrees. If I don't see/ can't interpret an explicit issue - what intervention would provide valuable feedback here - questions, guidance, validation, or giving a soft go-ahead with reminders of best practices?
4. In case the plan looks to be accurate - based on the context, can I remind the agent of how to continue, what not to forget, or should I soften and step back for the agent to continue its work? What's the most helpful thing I can do right now?"""

# === Fallback Response ===

FALLBACK_QUESTIONS = """
I can see you're thinking through your approach, which shows thoughtfulness:

1. Does this plan directly address what the user requested, or might it be solving a different problem?
2. Is there a simpler approach that would meet the user's needs?
3. What unstated assumptions might be limiting the thinking here?
4. How does this align with the user's original intent?
"""

# === Category Keywords ===

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Complex Solution Bias": ["complex", "complicated", "over-engineered", "complexity"],
    "Feature Creep": ["feature", "extra", "additional", "scope creep"],
    "Premature Implementation": ["premature", "early", "jumping", "too quick"],
    "Misalignment": ["misaligned", "wrong direction", "off target", "misunderstood"],
    "Overtooling": ["overtool", "too many tools", "unnecessary tools"],
}


def get_api_key(env_var: str) -> str | None:
    """Get API key from environment."""
    return os.environ.get(env_var)


def get_default_provider() -> str:
    """Get default LLM provider from environment."""
    return os.environ.get(DEFAULT_LLM_PROVIDER_ENV, "gemini")


def get_default_model() -> str | None:
    """Get default model from environment."""
    return os.environ.get(DEFAULT_MODEL_ENV)


def use_learning_history() -> bool:
    """Check if learning history should be used."""
    return os.environ.get(USE_LEARNING_HISTORY_ENV, "").lower() == "true"

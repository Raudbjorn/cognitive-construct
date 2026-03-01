"""
vibecheck - Python client library for metacognitive questioning and pattern learning

A tool for AI agent alignment and reflection. Monitors progress and questions
assumptions to mitigate Reasoning Lock-In.

Usage:
    from vibecheck import VibeCheckClient, VibeCheckInput, vibe_check, vibe_learn

    # Client-based usage (recommended for multiple calls)
    client = VibeCheckClient()

    # Run a vibe check
    result = await client.vibe_check(VibeCheckInput(
        goal="Ship CPI v2.5 with zero regressions",
        plan="1) Write tests 2) Refactor 3) Deploy",
        progress="Finished step 1",
        uncertainties=["uncertain about deployment"],
    ))

    if result.is_ok():
        print(result.value.questions)
    else:
        print(f"Error: {result.error.message}")

    # With model override
    from vibecheck import ModelOverride

    result = await client.vibe_check(VibeCheckInput(
        goal="Implement feature X",
        plan="Design -> Implement -> Test",
        model_override=ModelOverride(provider="anthropic", model="claude-3-opus"),
    ))

    # Learn from mistakes
    from vibecheck import VibeLearnInput, LearningCategory, LearningType

    result = await client.vibe_learn(VibeLearnInput(
        mistake="Skipped writing tests",
        category=LearningCategory.PREMATURE_IMPLEMENTATION,
        solution="Added regression tests",
        type=LearningType.MISTAKE,
    ))

    # Manage session constitution (rules)
    client.add_constitution_rule("session-123", "Always write tests first")
    rules = client.get_constitution_rules("session-123")
    client.reset_constitution_rules("session-123", ["Rule 1", "Rule 2"])

    # Convenience functions for one-off calls
    result = await vibe_check(
        goal="Quick task",
        plan="Simple plan",
        provider="gemini",
    )

    result = await vibe_learn(
        mistake="Forgot to handle edge case",
        category="Other",
        solution="Added edge case handling",
    )

Environment Variables:
    GEMINI_API_KEY: Google Gemini API key
    OPENAI_API_KEY: OpenAI API key
    OPENROUTER_API_KEY: OpenRouter API key
    ANTHROPIC_API_KEY: Anthropic API key
    ANTHROPIC_AUTH_TOKEN: Anthropic auth token (alternative to API key)
    ANTHROPIC_BASE_URL: Anthropic base URL (default: https://api.anthropic.com)
    DEFAULT_LLM_PROVIDER: Default provider (default: gemini)
    DEFAULT_MODEL: Default model for the provider
    USE_LEARNING_HISTORY: Set to "true" to include learning history in context
"""

from .client import VibeCheckClient, vibe_check, vibe_learn
from .result import Err, Ok, Result
from .types import (
    CategorySummary,
    ConstitutionResponse,
    LearningCategory,
    LearningEntry,
    LearningType,
    LLMProvider,
    ModelOverride,
    VibeCheckError,
    VibeCheckInput,
    VibeCheckResponse,
    VibeLearnInput,
    VibeLearnResponse,
)

__all__ = [
    # Client
    "VibeCheckClient",
    "vibe_check",
    "vibe_learn",
    # Input Types
    "VibeCheckInput",
    "VibeLearnInput",
    "ModelOverride",
    # Response Types
    "VibeCheckResponse",
    "VibeLearnResponse",
    "ConstitutionResponse",
    "LearningEntry",
    "CategorySummary",
    # Enums
    "LLMProvider",
    "LearningType",
    "LearningCategory",
    # Error Types
    "VibeCheckError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

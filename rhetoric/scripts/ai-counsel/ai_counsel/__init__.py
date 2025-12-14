"""
AI Counsel - Multi-model deliberative consensus library.

A Python library that enables true deliberative consensus between AI models.
Unlike parallel opinion gathering, models see each other's responses and refine
positions across multiple rounds of debate.

Example:
    from ai_counsel import AICounselClient, Participant, deliberate

    # Using client (recommended for multiple deliberations)
    client = AICounselClient(
        openrouter_api_key="sk-...",
        ollama_url="http://localhost:11434",
    )

    result = await client.deliberate(
        question="Should we use TypeScript or JavaScript for the frontend?",
        participants=[
            Participant(adapter="ollama", model="llama2"),
            Participant(adapter="openrouter", model="anthropic/claude-3.5-sonnet"),
        ],
        rounds=2,
    )

    if result.is_ok():
        print(result.value.summary.consensus)
    else:
        print(f"Error: {result.error.message}")

    # Using convenience function (for one-off deliberations)
    result = await deliberate(
        question="...",
        participants=[...],
    )
"""

__version__ = "2.0.0"

# Result types
from ai_counsel.result import Err, Ok, Result

# Public types (frozen dataclasses)
from ai_counsel.types import (
    ApiError,
    Contradiction,
    ConvergenceInfo,
    DeliberationRequest,
    DeliberationResult,
    Participant,
    RelatedDecision,
    RoundResponse,
    RoundVote,
    SearchResult,
    Summary,
    Timeline,
    ToolExecution,
    Vote,
    VotingResult,
)

# Client
from ai_counsel.client import (
    AICounselClient,
    deliberate,
    query_decisions,
)

__all__ = [
    # Version
    "__version__",
    # Result types
    "Ok",
    "Err",
    "Result",
    # Client
    "AICounselClient",
    "deliberate",
    "query_decisions",
    # Request/Response types
    "Participant",
    "DeliberationRequest",
    "DeliberationResult",
    "RoundResponse",
    "Summary",
    # Voting types
    "Vote",
    "RoundVote",
    "VotingResult",
    # Convergence types
    "ConvergenceInfo",
    # Tool types
    "ToolExecution",
    # Decision graph types
    "SearchResult",
    "Contradiction",
    "Timeline",
    "RelatedDecision",
    # Error types
    "ApiError",
]

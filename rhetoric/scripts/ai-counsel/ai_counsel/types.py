"""Public type definitions for AI Counsel client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# === Error Types ===


@dataclass(frozen=True, slots=True)
class ApiError:
    """Error from API operations."""

    error_type: str
    message: str
    details: dict[str, Any] | None = None

    @classmethod
    def from_exception(cls, e: Exception) -> ApiError:
        """Create ApiError from an exception."""
        return cls(
            error_type=type(e).__name__,
            message=str(e),
            details=None,
        )

    def __str__(self) -> str:
        if self.details:
            return f"{self.error_type}: {self.message} ({self.details})"
        return f"{self.error_type}: {self.message}"


# === Request Types ===


@dataclass(frozen=True, slots=True)
class Participant:
    """Deliberation participant configuration.

    Specifies which HTTP adapter and model to use for this participant.
    """

    adapter: Literal["ollama", "lmstudio", "openrouter"]
    model: str


@dataclass(frozen=True, slots=True)
class DeliberationRequest:
    """Request to start a deliberation.

    Attributes:
        question: The question or proposal to deliberate on (min 10 chars)
        participants: List of participant configurations (min 2)
        rounds: Number of deliberation rounds (1-5, default 2)
        mode: "quick" for fast consensus, "conference" for thorough debate
        context: Optional additional context for participants
        working_directory: Working directory for tool execution (default: cwd)
    """

    question: str
    participants: tuple[Participant, ...]
    rounds: int = 2
    mode: Literal["quick", "conference"] = "quick"
    context: str | None = None
    working_directory: str | None = None


# === Voting Types ===


@dataclass(frozen=True, slots=True)
class Vote:
    """Individual vote from a participant.

    Attributes:
        option: The voting option (e.g., "Option A", "Yes", "Approve")
        confidence: Confidence level in this vote (0.0-1.0)
        rationale: Explanation for this vote
        continue_debate: Whether participant wants to continue deliberating
    """

    option: str
    confidence: float
    rationale: str
    continue_debate: bool = True


@dataclass(frozen=True, slots=True)
class RoundVote:
    """Vote cast in a specific round.

    Attributes:
        round: Round number when vote was cast
        participant: Participant identifier
        vote: The vote cast by this participant
        timestamp: ISO 8601 timestamp when vote was cast
    """

    round: int
    participant: str
    vote: Vote
    timestamp: str


@dataclass(frozen=True, slots=True)
class VotingResult:
    """Aggregated voting results across all rounds.

    Attributes:
        final_tally: Final vote counts by option
        votes_by_round: All votes organized by round
        consensus_reached: Whether voting reached consensus
        winning_option: The winning option (None if tie or no consensus)
    """

    final_tally: dict[str, int]
    votes_by_round: tuple[RoundVote, ...]
    consensus_reached: bool
    winning_option: str | None


# === Convergence Types ===


@dataclass(frozen=True, slots=True)
class ConvergenceInfo:
    """Convergence detection metadata for deliberation rounds.

    Tracks similarity metrics between consecutive rounds to determine
    when models have reached consensus or stable disagreement.

    Attributes:
        detected: Whether convergence was detected
        detection_round: Round number where convergence occurred
        final_similarity: Final similarity score (0.0-1.0)
        status: Convergence status
        scores_by_round: Historical similarity scores
        per_participant_similarity: Latest similarity per participant
    """

    detected: bool
    detection_round: int | None
    final_similarity: float
    status: Literal[
        "converged",
        "diverging",
        "refining",
        "impasse",
        "max_rounds",
        "unanimous_consensus",
        "majority_decision",
        "tie",
        "unknown",
    ]
    scores_by_round: tuple[dict[str, Any], ...] = ()
    per_participant_similarity: dict[str, float] = field(default_factory=dict)


# === Response Types ===


@dataclass(frozen=True, slots=True)
class Summary:
    """Deliberation summary.

    Attributes:
        consensus: Overall consensus description
        key_agreements: Points of agreement
        key_disagreements: Points of disagreement
        final_recommendation: Final recommendation
    """

    consensus: str
    key_agreements: tuple[str, ...]
    key_disagreements: tuple[str, ...]
    final_recommendation: str


@dataclass(frozen=True, slots=True)
class RoundResponse:
    """Response from a single participant in a round.

    Attributes:
        round: Round number
        participant: Participant identifier
        response: The response text
        timestamp: ISO 8601 timestamp
    """

    round: int
    participant: str
    response: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Record of a tool execution during deliberation.

    Attributes:
        round: Round number when tool was executed
        participant: Participant who requested the tool
        tool_name: Name of the tool executed
        arguments: Arguments passed to the tool
        success: Whether execution succeeded
        output: Tool output or error message
        timestamp: ISO 8601 timestamp
    """

    round: int
    participant: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class DeliberationResult:
    """Complete deliberation result.

    Attributes:
        status: "complete", "partial", or "failed"
        mode: Mode used ("quick" or "conference")
        rounds_completed: Number of rounds completed
        participants: List of participant identifiers
        summary: Deliberation summary
        transcript_path: Path to full transcript
        full_debate: Full debate history
        convergence_info: Convergence detection information
        voting_result: Voting results if participants cast votes
        graph_context_summary: Summary of decision graph context used
        tool_executions: List of tool executions during deliberation
    """

    status: Literal["complete", "partial", "failed"]
    mode: str
    rounds_completed: int
    participants: tuple[str, ...]
    summary: Summary
    transcript_path: str
    full_debate: tuple[RoundResponse, ...]
    convergence_info: ConvergenceInfo | None = None
    voting_result: VotingResult | None = None
    graph_context_summary: str | None = None
    tool_executions: tuple[ToolExecution, ...] = ()


# === Decision Graph Types ===


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result from decision graph search.

    Attributes:
        decision_id: Unique decision identifier
        question: The original question
        consensus: The consensus reached
        participants: List of participant identifiers
        score: Similarity score (0.0-1.0)
        timestamp: ISO 8601 timestamp
    """

    decision_id: str
    question: str
    consensus: str
    participants: tuple[str, ...]
    score: float
    timestamp: str


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Detected contradiction between decisions.

    Attributes:
        decision_id_1: First decision ID
        decision_id_2: Second decision ID
        question_1: First question
        question_2: Second question
        severity: Contradiction severity (0.0-1.0)
        description: Description of the contradiction
    """

    decision_id_1: str
    decision_id_2: str
    question_1: str
    question_2: str
    severity: float
    description: str


@dataclass(frozen=True, slots=True)
class RelatedDecision:
    """A decision related to another in a timeline.

    Attributes:
        decision_id: Related decision ID
        question: The question
        consensus: The consensus
        similarity: Similarity score
        timestamp: ISO 8601 timestamp
    """

    decision_id: str
    question: str
    consensus: str
    similarity: float
    timestamp: str


@dataclass(frozen=True, slots=True)
class Timeline:
    """Evolution timeline for a decision.

    Attributes:
        decision_id: Decision ID
        question: The question
        consensus: The consensus
        status: Convergence status
        participants: Participant identifiers
        related_decisions: Related decisions in chronological order
    """

    decision_id: str
    question: str
    consensus: str
    status: str
    participants: tuple[str, ...]
    related_decisions: tuple[RelatedDecision, ...] = ()

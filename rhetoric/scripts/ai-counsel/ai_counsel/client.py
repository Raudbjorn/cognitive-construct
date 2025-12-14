"""AI Counsel client for multi-model deliberative consensus."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_counsel.result import Err, Ok, Result
from ai_counsel.types import (
    ApiError,
    ConvergenceInfo,
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

if TYPE_CHECKING:
    from ai_counsel._internal.config import ClientConfig

logger = logging.getLogger(__name__)


def _add_project_to_path() -> None:
    """Add project root to path for importing existing modules."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


class AICounselClient:
    """Client for AI Counsel multi-model deliberative consensus.

    Provides async methods for running deliberations and querying the decision graph.
    All public methods return Result types instead of raising exceptions.

    Example:
        client = AICounselClient(
            openrouter_api_key="sk-...",
            ollama_url="http://localhost:11434",
        )

        result = await client.deliberate(
            question="Should we use TypeScript or JavaScript?",
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
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        ollama_url: str | None = None,
        lmstudio_url: str | None = None,
        openrouter_api_key: str | None = None,
        openrouter_url: str = "https://openrouter.ai/api/v1",
        openai_api_key: str | None = None,
        openai_url: str = "https://api.openai.com",
        anthropic_api_key: str | None = None,
        anthropic_url: str = "https://api.anthropic.com",
        decision_graph_path: str | Path | None = None,
        enable_decision_graph: bool = True,
        enable_transcripts: bool = True,
        transcripts_dir: str = "transcripts",
        timeout: int = 300,
        working_directory: str | Path | None = None,
    ):
        """Initialize AI Counsel client.

        Configuration priority:
        1. Explicit constructor parameters
        2. Environment variables (OLLAMA_URL, LMSTUDIO_URL, OPENROUTER_API_KEY, etc.)
        3. Config file (if provided)
        4. Defaults

        Args:
            config_path: Path to YAML config file (optional)
            ollama_url: Ollama API URL (or OLLAMA_URL env var)
            lmstudio_url: LM Studio API URL (or LMSTUDIO_URL env var)
            openrouter_api_key: OpenRouter API key (or OPENROUTER_API_KEY env var)
            openrouter_url: OpenRouter API URL
            openai_api_key: OpenAI API key (or OPENAI_API_KEY env var)
            openai_url: OpenAI API URL
            anthropic_api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            anthropic_url: Anthropic API URL
            decision_graph_path: Path to decision graph SQLite DB
            enable_decision_graph: Enable decision graph memory
            enable_transcripts: Enable transcript generation
            transcripts_dir: Directory for transcript files
            timeout: Default timeout for adapter calls
            working_directory: Default working directory for tools
        """
        _add_project_to_path()

        self._config = self._build_config(
            config_path=config_path,
            ollama_url=ollama_url,
            lmstudio_url=lmstudio_url,
            openrouter_api_key=openrouter_api_key,
            openrouter_url=openrouter_url,
            openai_api_key=openai_api_key,
            openai_url=openai_url,
            anthropic_api_key=anthropic_api_key,
            anthropic_url=anthropic_url,
            decision_graph_path=decision_graph_path,
            enable_decision_graph=enable_decision_graph,
            timeout=timeout,
        )

        self._adapters = self._create_adapters()
        self._working_directory = (
            str(working_directory) if working_directory else os.getcwd()
        )
        self._enable_transcripts = enable_transcripts
        self._transcripts_dir = transcripts_dir

        # Lazy initialization for engine
        self._engine: Any | None = None
        self._query_engine: Any | None = None

    def _build_config(
        self,
        config_path: str | Path | None,
        ollama_url: str | None,
        lmstudio_url: str | None,
        openrouter_api_key: str | None,
        openrouter_url: str,
        openai_api_key: str | None,
        openai_url: str,
        anthropic_api_key: str | None,
        anthropic_url: str,
        decision_graph_path: str | Path | None,
        enable_decision_graph: bool,
        timeout: int,
    ) -> ClientConfig:
        """Build client configuration from various sources."""
        from ai_counsel._internal.config import (
            ClientConfig,
            DecisionGraphConfig,
            HTTPAdapterConfig,
            load_config,
        )

        # Start with config file if provided
        if config_path:
            try:
                config = load_config(config_path)
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                config = ClientConfig()
        else:
            config = ClientConfig()

        # Override with explicit parameters
        adapters = dict(config.adapters) if config.adapters else {}

        # Resolve from env vars if not provided
        resolved_ollama = ollama_url or os.getenv("OLLAMA_URL")
        resolved_lmstudio = lmstudio_url or os.getenv("LMSTUDIO_URL")
        resolved_openrouter_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        resolved_openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        resolved_anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if resolved_ollama:
            adapters["ollama"] = HTTPAdapterConfig(
                base_url=resolved_ollama,
                timeout=timeout,
            )

        if resolved_lmstudio:
            adapters["lmstudio"] = HTTPAdapterConfig(
                base_url=resolved_lmstudio,
                timeout=timeout,
            )

        if resolved_openrouter_key:
            adapters["openrouter"] = HTTPAdapterConfig(
                base_url=openrouter_url,
                api_key=resolved_openrouter_key,
                timeout=timeout,
            )

        if resolved_openai_key:
            adapters["openai"] = HTTPAdapterConfig(
                base_url=openai_url,
                api_key=resolved_openai_key,
                timeout=timeout,
            )

        if resolved_anthropic_key:
            adapters["anthropic"] = HTTPAdapterConfig(
                base_url=anthropic_url,
                api_key=resolved_anthropic_key,
                timeout=timeout,
            )

        # Configure decision graph
        decision_graph = None
        if enable_decision_graph:
            db_path = str(decision_graph_path) if decision_graph_path else "decision_graph.db"
            decision_graph = DecisionGraphConfig(
                enabled=True,
                db_path=db_path,
            )

        return ClientConfig(
            adapters=adapters,
            defaults=config.defaults,
            storage=config.storage,
            deliberation=config.deliberation,
            decision_graph=decision_graph,
        )

    def _create_adapters(self) -> dict[str, Any]:
        """Create HTTP adapter instances from config."""
        from ai_counsel._internal.adapters import create_adapter

        adapters = {}
        for name, adapter_config in self._config.adapters.items():
            try:
                adapters[name] = create_adapter(name, adapter_config)
                logger.info(f"Created adapter: {name} ({adapter_config.base_url})")
            except Exception as e:
                logger.warning(f"Failed to create adapter {name}: {e}")

        return adapters

    def _get_engine(self) -> Any:
        """Get or create the deliberation engine."""
        if self._engine is None:
            # Import from existing location
            from deliberation.engine import DeliberationEngine
            from deliberation.transcript import TranscriptManager

            transcript_manager = None
            if self._enable_transcripts:
                transcript_manager = TranscriptManager(
                    output_dir=self._transcripts_dir
                )

            # Build config object that engine expects
            self._engine = DeliberationEngine(
                adapters=self._adapters,
                transcript_manager=transcript_manager,
                config=self._build_engine_config(),
            )

        return self._engine

    def _build_engine_config(self) -> Any:
        """Build config object in format expected by engine."""
        # The engine expects the old Config format
        # Create a compatible object
        from types import SimpleNamespace

        config = SimpleNamespace()
        config.defaults = self._config.defaults
        config.deliberation = self._config.deliberation
        config.decision_graph = self._config.decision_graph
        config.storage = self._config.storage

        return config

    async def deliberate(
        self,
        question: str,
        participants: list[Participant],
        *,
        rounds: int = 2,
        mode: str = "quick",
        context: str | None = None,
        working_directory: str | None = None,
    ) -> Result[DeliberationResult, ApiError]:
        """Execute multi-model deliberation.

        Args:
            question: The question or proposal to deliberate
            participants: List of Participant configurations
            rounds: Number of deliberation rounds (1-5)
            mode: "quick" (single round) or "conference" (multi-round)
            context: Additional context for deliberation
            working_directory: Working directory for tool execution

        Returns:
            Result containing DeliberationResult or ApiError
        """
        if not participants or len(participants) < 2:
            return Err(ApiError(
                error_type="ValidationError",
                message="At least 2 participants required",
            ))

        if not question or len(question) < 10:
            return Err(ApiError(
                error_type="ValidationError",
                message="Question must be at least 10 characters",
            ))

        # Validate all participants have available adapters
        for p in participants:
            if p.adapter not in self._adapters:
                available = ", ".join(sorted(self._adapters.keys()))
                return Err(ApiError(
                    error_type="ConfigurationError",
                    message=f"Adapter '{p.adapter}' not configured. Available: {available}",
                ))

        # Build request in format expected by engine
        from models.schema import DeliberateRequest as InternalRequest
        from models.schema import Participant as InternalParticipant

        internal_participants = [
            InternalParticipant(cli=p.adapter, model=p.model)
            for p in participants
        ]

        internal_request = InternalRequest(
            question=question,
            participants=internal_participants,
            rounds=min(max(1, rounds), 5),  # Clamp to 1-5
            mode=mode if mode in ("quick", "conference") else "quick",
            context=context,
            working_directory=working_directory or self._working_directory,
        )

        try:
            engine = self._get_engine()
            internal_result = await engine.execute(internal_request)
            return Ok(self._convert_result(internal_result))
        except Exception as e:
            logger.error(f"Deliberation failed: {e}", exc_info=True)
            return Err(ApiError.from_exception(e))

    def _convert_result(self, internal: Any) -> DeliberationResult:
        """Convert internal Pydantic result to frozen dataclass."""
        # Convert summary
        summary = Summary(
            consensus=internal.summary.consensus,
            key_agreements=tuple(internal.summary.key_agreements),
            key_disagreements=tuple(internal.summary.key_disagreements),
            final_recommendation=internal.summary.final_recommendation,
        )

        # Convert full_debate
        full_debate = tuple(
            RoundResponse(
                round=r.round,
                participant=r.participant,
                response=r.response,
                timestamp=r.timestamp,
            )
            for r in internal.full_debate
        )

        # Convert convergence_info
        convergence_info = None
        if internal.convergence_info:
            ci = internal.convergence_info
            convergence_info = ConvergenceInfo(
                detected=ci.detected,
                detection_round=ci.detection_round,
                final_similarity=ci.final_similarity,
                status=ci.status,
                scores_by_round=tuple(ci.scores_by_round) if ci.scores_by_round else (),
                per_participant_similarity=dict(ci.per_participant_similarity) if ci.per_participant_similarity else {},
            )

        # Convert voting_result
        voting_result = None
        if internal.voting_result:
            vr = internal.voting_result
            votes_by_round = tuple(
                RoundVote(
                    round=rv.round,
                    participant=rv.participant,
                    vote=Vote(
                        option=rv.vote.option,
                        confidence=rv.vote.confidence,
                        rationale=rv.vote.rationale,
                        continue_debate=rv.vote.continue_debate,
                    ),
                    timestamp=rv.timestamp,
                )
                for rv in vr.votes_by_round
            )
            voting_result = VotingResult(
                final_tally=dict(vr.final_tally),
                votes_by_round=votes_by_round,
                consensus_reached=vr.consensus_reached,
                winning_option=vr.winning_option,
            )

        # Convert tool_executions
        tool_executions = ()
        if internal.tool_executions:
            tool_executions = tuple(
                ToolExecution(
                    round=te.round_number,
                    participant=te.requested_by,
                    tool_name=te.request.name,
                    arguments=dict(te.request.arguments),
                    success=te.result.success,
                    output=te.result.output or te.result.error or "",
                    timestamp=te.timestamp,
                )
                for te in internal.tool_executions
            )

        return DeliberationResult(
            status=internal.status,
            mode=internal.mode,
            rounds_completed=internal.rounds_completed,
            participants=tuple(internal.participants),
            summary=summary,
            transcript_path=internal.transcript_path,
            full_debate=full_debate,
            convergence_info=convergence_info,
            voting_result=voting_result,
            graph_context_summary=internal.graph_context_summary,
            tool_executions=tool_executions,
        )

    async def query_decisions(
        self,
        query: str,
        *,
        limit: int = 5,
        threshold: float = 0.6,
    ) -> Result[list[SearchResult], ApiError]:
        """Search for similar past deliberations.

        Args:
            query: Search query text
            limit: Maximum results to return
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            Result containing list of SearchResult or ApiError
        """
        if not self._config.decision_graph or not self._config.decision_graph.enabled:
            return Err(ApiError(
                error_type="ConfigurationError",
                message="Decision graph is not enabled",
            ))

        try:
            engine = self._get_query_engine()
            if engine is None:
                return Err(ApiError(
                    error_type="ConfigurationError",
                    message="Query engine not available",
                ))

            results = engine.search_similar(query, limit=limit, threshold=threshold)
            converted = [
                SearchResult(
                    decision_id=r.decision_id,
                    question=r.question,
                    consensus=r.consensus,
                    participants=tuple(r.participants),
                    score=r.score,
                    timestamp=r.timestamp,
                )
                for r in results
            ]
            return Ok(converted)
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return Err(ApiError.from_exception(e))

    def _get_query_engine(self) -> Any | None:
        """Get or create the query engine."""
        if self._query_engine is None:
            if not self._config.decision_graph or not self._config.decision_graph.enabled:
                return None

            try:
                from decision_graph.storage import DecisionGraphStorage
                from deliberation.query_engine import QueryEngine

                db_path = self._config.decision_graph.db_path
                storage = DecisionGraphStorage(db_path)
                self._query_engine = QueryEngine(storage, self._config.decision_graph)
            except Exception as e:
                logger.warning(f"Failed to initialize query engine: {e}")
                return None

        return self._query_engine

    async def trace_decision(
        self,
        decision_id: str,
        include_related: bool = False,
    ) -> Result[Timeline, ApiError]:
        """Trace evolution of a specific decision.

        Args:
            decision_id: Decision ID to trace
            include_related: Include related decisions

        Returns:
            Result containing Timeline or ApiError
        """
        try:
            engine = self._get_query_engine()
            if engine is None:
                return Err(ApiError(
                    error_type="ConfigurationError",
                    message="Query engine not available",
                ))

            result = engine.trace_evolution(decision_id, include_related=include_related)
            if result is None:
                return Err(ApiError(
                    error_type="NotFoundError",
                    message=f"Decision not found: {decision_id}",
                ))

            related = tuple(
                RelatedDecision(
                    decision_id=r["decision_id"],
                    question=r["question"],
                    consensus=r["consensus"],
                    similarity=r.get("similarity", 0.0),
                    timestamp=r.get("timestamp", ""),
                )
                for r in result.get("related_decisions", [])
            )

            return Ok(Timeline(
                decision_id=result["decision_id"],
                question=result["question"],
                consensus=result["consensus"],
                status=result.get("status", "unknown"),
                participants=tuple(result.get("participants", [])),
                related_decisions=related,
            ))
        except Exception as e:
            logger.error(f"Trace failed: {e}", exc_info=True)
            return Err(ApiError.from_exception(e))

    async def close(self) -> None:
        """Close client and release resources."""
        self._engine = None
        self._query_engine = None


# Convenience functions


async def deliberate(
    question: str,
    participants: list[Participant],
    *,
    rounds: int = 2,
    mode: str = "quick",
    context: str | None = None,
    working_directory: str | None = None,
    **client_kwargs: Any,
) -> Result[DeliberationResult, ApiError]:
    """Convenience function for one-off deliberations.

    Creates a temporary client, executes deliberation, and closes.
    For multiple deliberations, prefer creating an AICounselClient instance.

    Args:
        question: The question or proposal to deliberate
        participants: List of Participant configurations
        rounds: Number of deliberation rounds (1-5)
        mode: "quick" (single round) or "conference" (multi-round)
        context: Additional context for deliberation
        working_directory: Working directory for tool execution
        **client_kwargs: Additional arguments for AICounselClient

    Returns:
        Result containing DeliberationResult or ApiError
    """
    client = AICounselClient(**client_kwargs)
    try:
        return await client.deliberate(
            question=question,
            participants=participants,
            rounds=rounds,
            mode=mode,
            context=context,
            working_directory=working_directory,
        )
    finally:
        await client.close()


async def query_decisions(
    query: str,
    *,
    limit: int = 5,
    threshold: float = 0.6,
    **client_kwargs: Any,
) -> Result[list[SearchResult], ApiError]:
    """Convenience function for querying decisions.

    Creates a temporary client, executes query, and closes.

    Args:
        query: Search query text
        limit: Maximum results to return
        threshold: Minimum similarity score (0.0-1.0)
        **client_kwargs: Additional arguments for AICounselClient

    Returns:
        Result containing list of SearchResult or ApiError
    """
    client = AICounselClient(**client_kwargs)
    try:
        return await client.query_decisions(query, limit=limit, threshold=threshold)
    finally:
        await client.close()

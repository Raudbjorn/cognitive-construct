"""Vibe check client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import (
    FALLBACK_QUESTIONS,
    METACOGNITIVE_SYSTEM_PROMPT,
    get_default_model,
    get_default_provider,
    use_learning_history,
)
from .constitution import get_constitution, reset_constitution, update_constitution
from .providers import get_provider
from .result import Err, Ok, Result
from .state import add_to_history, ensure_initialized, get_history_summary
from .storage import (
    add_learning_entry,
    enforce_one_sentence,
    get_learning_category_summary,
    get_learning_context_text,
    get_learning_entries,
    is_similar,
    normalize_category,
)
from .types import (
    CategorySummary,
    ConstitutionResponse,
    LearningCategory,
    LearningType,
    LLMProvider as LLMProviderEnum,
    ModelOverride,
    VibeCheckError,
    VibeCheckInput,
    VibeCheckResponse,
    VibeLearnInput,
    VibeLearnResponse,
)


@dataclass
class VibeCheckClient:
    """Client for vibe check - metacognitive questioning for AI alignment.

    Usage:
        client = VibeCheckClient()

        # Basic vibe check
        result = await client.vibe_check(VibeCheckInput(
            goal="Ship CPI v2.5 with zero regressions",
            plan="1) Write tests 2) Refactor 3) Deploy",
        ))

        if result.is_ok():
            print(result.value.questions)
        else:
            print(f"Error: {result.error.message}")

        # With model override
        result = await client.vibe_check(VibeCheckInput(
            goal="Implement feature X",
            plan="Design -> Implement -> Test",
            model_override=ModelOverride(provider="anthropic", model="claude-3-opus"),
        ))

        # Learn from mistakes
        result = await client.vibe_learn(VibeLearnInput(
            mistake="Skipped writing tests",
            category=LearningCategory.PREMATURE_IMPLEMENTATION,
            solution="Added regression tests",
        ))

        # Manage constitution
        client.add_constitution_rule("session-123", "Always write tests first")
        rules = client.get_constitution_rules("session-123")
    """

    default_provider: str | None = None
    default_model: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        """Initialize defaults from environment."""
        if not self.default_provider:
            self.default_provider = get_default_provider()
        if not self.default_model:
            self.default_model = get_default_model()

    async def vibe_check(
        self,
        input_data: VibeCheckInput,
    ) -> Result[VibeCheckResponse, VibeCheckError]:
        """Run a vibe check to get metacognitive questions.

        Args:
            input_data: The vibe check input with goal, plan, and optional context

        Returns:
            Result containing VibeCheckResponse with questions or VibeCheckError
        """
        # Validate required fields
        if not input_data.goal:
            return Err(VibeCheckError(message="Goal is required", code="INVALID_INPUT"))
        if not input_data.plan:
            return Err(VibeCheckError(message="Plan is required", code="INVALID_INPUT"))

        # Ensure history is loaded
        ensure_initialized()

        # Get provider and model
        provider_name = (
            input_data.model_override.provider
            if input_data.model_override and input_data.model_override.provider
            else self.default_provider
        )
        if isinstance(provider_name, LLMProviderEnum):
            provider_name = provider_name.value

        model = (
            input_data.model_override.model
            if input_data.model_override and input_data.model_override.model
            else self.default_model
        )

        # Build context
        history_summary = get_history_summary(input_data.session_id)
        learning_context = get_learning_context_text() if use_learning_history() else ""
        rules = get_constitution(input_data.session_id) if input_data.session_id else []
        constitution_block = (
            f"\nConstitution:\n" + "\n".join(f"- {r}" for r in rules)
            if rules
            else ""
        )

        # Build prompt
        context_section = f"""CONTEXT:
History Context: {history_summary or 'None'}
{f'Learning Context:\n{learning_context}' if learning_context else ''}
Goal: {input_data.goal}
Plan: {input_data.plan}
Progress: {input_data.progress or 'None'}
Uncertainties: {', '.join(input_data.uncertainties) if input_data.uncertainties else 'None'}
Task Context: {input_data.task_context or 'None'}
User Prompt: {input_data.user_prompt or 'None'}{constitution_block}"""

        # Get provider and generate
        try:
            provider = get_provider(provider_name or "gemini")
            result = await provider.generate(
                prompt=context_section,
                system_prompt=METACOGNITIVE_SYSTEM_PROMPT,
                model=model,
            )

            if result.is_err():
                # Return fallback questions on provider error
                add_to_history(input_data.session_id, input_data, FALLBACK_QUESTIONS)
                return Ok(VibeCheckResponse(questions=FALLBACK_QUESTIONS))

            questions = result.value.text
            add_to_history(input_data.session_id, input_data, questions)
            return Ok(VibeCheckResponse(questions=questions))

        except Exception as e:
            # Return fallback questions on any error
            add_to_history(input_data.session_id, input_data, FALLBACK_QUESTIONS)
            return Ok(VibeCheckResponse(questions=FALLBACK_QUESTIONS))

    async def vibe_learn(
        self,
        input_data: VibeLearnInput,
    ) -> Result[VibeLearnResponse, VibeCheckError]:
        """Record a learning entry for pattern recognition.

        Args:
            input_data: The learning input with mistake, category, and optional solution

        Returns:
            Result containing VibeLearnResponse or VibeCheckError
        """
        # Validate required fields
        if not input_data.mistake:
            return Err(VibeCheckError(message="Mistake description is required", code="INVALID_INPUT"))
        if not input_data.category:
            return Err(VibeCheckError(message="Category is required", code="INVALID_INPUT"))

        entry_type = input_data.type
        if entry_type != LearningType.PREFERENCE and not input_data.solution:
            return Err(
                VibeCheckError(
                    message="Solution is required for this entry type",
                    code="INVALID_INPUT",
                )
            )

        # Enforce single-sentence constraints
        mistake = enforce_one_sentence(input_data.mistake)
        solution = enforce_one_sentence(input_data.solution) if input_data.solution else None

        # Normalize category
        category_str = (
            input_data.category.value
            if isinstance(input_data.category, LearningCategory)
            else input_data.category
        )
        category = normalize_category(category_str)

        # Check for similar existing entry
        existing = get_learning_entries().get(category, [])
        already_known = any(is_similar(e.mistake, mistake) for e in existing)

        # Add if new
        if not already_known:
            add_learning_entry(mistake, category, solution, entry_type)

        # Get category summaries
        category_summary = get_learning_category_summary()

        # Find current tally for this category
        category_data = next((m for m in category_summary if m.category == category), None)
        current_tally = category_data.count if category_data else 1

        # Get top 3 categories
        top_categories = category_summary[:3]

        return Ok(
            VibeLearnResponse(
                added=not already_known,
                already_known=already_known,
                current_tally=current_tally,
                top_categories=top_categories,
            )
        )

    def add_constitution_rule(self, session_id: str, rule: str) -> list[str]:
        """Add a rule to a session's constitution.

        Args:
            session_id: The session identifier
            rule: The rule to add

        Returns:
            The updated list of rules
        """
        if not session_id or not rule:
            raise ValueError("session_id and rule are required")

        update_constitution(session_id, rule)
        return get_constitution(session_id)

    def reset_constitution_rules(self, session_id: str, rules: list[str]) -> list[str]:
        """Replace all rules for a session.

        Args:
            session_id: The session identifier
            rules: The new list of rules

        Returns:
            The updated list of rules
        """
        if not session_id:
            raise ValueError("session_id is required")
        if not isinstance(rules, list):
            raise ValueError("rules must be a list of strings")

        reset_constitution(session_id, rules)
        return get_constitution(session_id)

    def get_constitution_rules(self, session_id: str) -> list[str]:
        """Get rules for a session.

        Args:
            session_id: The session identifier

        Returns:
            The list of rules
        """
        if not session_id:
            raise ValueError("session_id is required")

        return get_constitution(session_id)


# === Convenience Functions ===


async def vibe_check(
    goal: str,
    plan: str,
    user_prompt: str | None = None,
    progress: str | None = None,
    uncertainties: list[str] | None = None,
    task_context: str | None = None,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Result[VibeCheckResponse, VibeCheckError]:
    """Run a vibe check (convenience function).

    Creates a client for single use. For multiple checks,
    prefer creating a VibeCheckClient instance.
    """
    client = VibeCheckClient(default_provider=provider, default_model=model)

    model_override = None
    if provider or model:
        model_override = ModelOverride(provider=provider, model=model)

    return await client.vibe_check(
        VibeCheckInput(
            goal=goal,
            plan=plan,
            user_prompt=user_prompt,
            progress=progress,
            uncertainties=uncertainties or [],
            task_context=task_context,
            session_id=session_id,
            model_override=model_override,
        )
    )


async def vibe_learn(
    mistake: str,
    category: str | LearningCategory,
    solution: str | None = None,
    entry_type: LearningType = LearningType.MISTAKE,
    session_id: str | None = None,
) -> Result[VibeLearnResponse, VibeCheckError]:
    """Record a learning entry (convenience function).

    Creates a client for single use. For multiple entries,
    prefer creating a VibeCheckClient instance.
    """
    client = VibeCheckClient()
    return await client.vibe_learn(
        VibeLearnInput(
            mistake=mistake,
            category=category,
            solution=solution,
            type=entry_type,
            session_id=session_id,
        )
    )

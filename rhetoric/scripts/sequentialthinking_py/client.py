"""Sequential thinking client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .result import Err, Ok, Result
from .types import ThinkingError, ThoughtData, ThoughtInput, ThoughtResponse


@dataclass
class SequentialThinkingClient:
    """Client for sequential thinking / chain-of-thought reasoning.

    Usage:
        client = SequentialThinkingClient()

        # Process first thought
        result = client.process_thought(ThoughtInput(
            thought="Let me analyze this problem step by step...",
            thought_number=1,
            total_thoughts=5,
            next_thought_needed=True,
        ))

        if result.is_ok():
            response = result.value
            print(f"Thought {response.thought_number}/{response.total_thoughts}")

        # Process a revision
        result = client.process_thought(ThoughtInput(
            thought="Actually, let me reconsider...",
            thought_number=2,
            total_thoughts=5,
            next_thought_needed=True,
            is_revision=True,
            revises_thought=1,
        ))

        # Branch into alternative path
        result = client.process_thought(ThoughtInput(
            thought="Exploring alternative approach...",
            thought_number=3,
            total_thoughts=6,
            next_thought_needed=True,
            branch_from_thought=1,
            branch_id="alt-approach",
        ))

        # Get current state
        history = client.get_history()
        branches = client.get_branches()
    """

    disable_logging: bool = False
    _thought_history: list[ThoughtData] = field(default_factory=list, init=False, repr=False)
    _branches: dict[str, list[ThoughtData]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize from environment if not explicitly set."""
        if not self.disable_logging:
            env_val = os.environ.get("DISABLE_THOUGHT_LOGGING", "").lower()
            self.disable_logging = env_val == "true"

    def process_thought(self, input_data: ThoughtInput) -> Result[ThoughtResponse, ThinkingError]:
        """Process a single thought step.

        Args:
            input_data: The thought input containing the thought text and metadata

        Returns:
            Result containing ThoughtResponse on success or ThinkingError on failure
        """
        # Validate required fields
        if not input_data.thought:
            return Err(ThinkingError(message="Thought text is required", code="INVALID_INPUT"))

        if input_data.thought_number < 1:
            return Err(
                ThinkingError(message="Thought number must be >= 1", code="INVALID_INPUT")
            )

        if input_data.total_thoughts < 1:
            return Err(
                ThinkingError(message="Total thoughts must be >= 1", code="INVALID_INPUT")
            )

        # Adjust total_thoughts if thought_number exceeds it
        adjusted_total = max(input_data.total_thoughts, input_data.thought_number)

        # Create thought data
        thought_data = ThoughtData(
            thought=input_data.thought,
            thought_number=input_data.thought_number,
            total_thoughts=adjusted_total,
            next_thought_needed=input_data.next_thought_needed,
            is_revision=input_data.is_revision,
            revises_thought=input_data.revises_thought,
            branch_from_thought=input_data.branch_from_thought,
            branch_id=input_data.branch_id,
            needs_more_thoughts=input_data.needs_more_thoughts,
        )

        # Add to history
        self._thought_history.append(thought_data)

        # Handle branching
        if input_data.branch_from_thought and input_data.branch_id:
            if input_data.branch_id not in self._branches:
                self._branches[input_data.branch_id] = []
            self._branches[input_data.branch_id].append(thought_data)

        # Format and log if enabled
        if not self.disable_logging:
            formatted = self._format_thought(thought_data)
            print(formatted)

        return Ok(
            ThoughtResponse(
                thought_number=input_data.thought_number,
                total_thoughts=adjusted_total,
                next_thought_needed=input_data.next_thought_needed,
                branches=list(self._branches.keys()),
                thought_history_length=len(self._thought_history),
            )
        )

    def get_history(self) -> list[ThoughtData]:
        """Get the full thought history."""
        return list(self._thought_history)

    def get_branches(self) -> dict[str, list[ThoughtData]]:
        """Get all branches and their thoughts."""
        return {k: list(v) for k, v in self._branches.items()}

    def get_branch(self, branch_id: str) -> list[ThoughtData] | None:
        """Get thoughts for a specific branch."""
        branch = self._branches.get(branch_id)
        return list(branch) if branch else None

    def clear(self) -> None:
        """Clear all thought history and branches."""
        self._thought_history.clear()
        self._branches.clear()

    def _format_thought(self, thought_data: ThoughtData) -> str:
        """Format a thought for display."""
        prefix = ""
        context = ""

        if thought_data.is_revision:
            prefix = "[Revision]"
            context = f" (revising thought {thought_data.revises_thought})"
        elif thought_data.branch_from_thought:
            prefix = "[Branch]"
            context = f" (from thought {thought_data.branch_from_thought}, ID: {thought_data.branch_id})"
        else:
            prefix = "[Thought]"
            context = ""

        header = f"{prefix} {thought_data.thought_number}/{thought_data.total_thoughts}{context}"
        width = max(len(header), min(len(thought_data.thought), 80)) + 4
        border = "-" * width

        return f"""
+{border}+
| {header.ljust(width - 2)} |
+{border}+
| {thought_data.thought[:width - 4].ljust(width - 2)} |
+{border}+"""


def process_thought(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    next_thought_needed: bool,
    is_revision: bool | None = None,
    revises_thought: int | None = None,
    branch_from_thought: int | None = None,
    branch_id: str | None = None,
    needs_more_thoughts: bool | None = None,
) -> Result[ThoughtResponse, ThinkingError]:
    """Process a single thought (convenience function).

    Creates a new client for single use. For multiple thoughts
    in a chain, prefer creating a SequentialThinkingClient instance.
    """
    client = SequentialThinkingClient(disable_logging=True)
    return client.process_thought(
        ThoughtInput(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from_thought=branch_from_thought,
            branch_id=branch_id,
            needs_more_thoughts=needs_more_thoughts,
        )
    )

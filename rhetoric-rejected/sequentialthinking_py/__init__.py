"""
sequentialthinking - Python client library for sequential thinking / chain-of-thought reasoning

Usage:
    from sequentialthinking import SequentialThinkingClient, ThoughtInput

    # Create client instance for a thinking session
    client = SequentialThinkingClient()

    # Process thoughts in sequence
    result = client.process_thought(ThoughtInput(
        thought="Let me break down this problem...",
        thought_number=1,
        total_thoughts=5,
        next_thought_needed=True,
    ))

    if result.is_ok():
        response = result.value
        print(f"Thought {response.thought_number}/{response.total_thoughts}")
        print(f"Continue: {response.next_thought_needed}")
    else:
        print(f"Error: {result.error.message}")

    # Support for revisions
    result = client.process_thought(ThoughtInput(
        thought="Actually, reconsidering step 1...",
        thought_number=2,
        total_thoughts=5,
        next_thought_needed=True,
        is_revision=True,
        revises_thought=1,
    ))

    # Support for branching
    result = client.process_thought(ThoughtInput(
        thought="Exploring alternative approach...",
        thought_number=3,
        total_thoughts=6,
        next_thought_needed=True,
        branch_from_thought=1,
        branch_id="alternative",
    ))

    # Get state
    history = client.get_history()
    branches = client.get_branches()

    # Convenience function for one-off thoughts
    from sequentialthinking import process_thought

    result = process_thought(
        thought="Quick analysis...",
        thought_number=1,
        total_thoughts=1,
        next_thought_needed=False,
    )
"""

from .client import SequentialThinkingClient, process_thought
from .result import Err, Ok, Result
from .types import ThinkingError, ThoughtData, ThoughtInput, ThoughtResponse

__all__ = [
    # Client
    "SequentialThinkingClient",
    "process_thought",
    # Types
    "ThoughtInput",
    "ThoughtResponse",
    "ThoughtData",
    "ThinkingError",
    # Result
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

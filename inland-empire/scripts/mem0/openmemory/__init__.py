"""
openmemory - Pure Python client library for Mem0 Cloud API

Usage:
    from openmemory import Mem0Client, add, search

    # Client-based usage (recommended for multiple calls)
    client = Mem0Client(api_key="...")  # or set MEM0_API_KEY env var

    result = await client.add("User likes Italian food", user_id="user123")
    if result.is_ok():
        for item in result.value.results:
            print(f"{item.event}: {item.memory}")
    else:
        print(f"Error: {result.error.message}")

    # Convenience function (for one-off calls)
    result = await search("food preferences", user_id="user123")
"""

from .client import (
    Mem0Client,
    add,
    search,
)
from .result import Err, Ok, Result
from .types import (
    # Error
    ApiError,
    # Batch types
    BatchDeleteResponse,
    BatchMemoryDelete,
    BatchMemoryUpdate,
    BatchUpdateResponse,
    # Entity types
    EntitiesResponse,
    Entity,
    EntityType,
    # Export types
    ExportCreateResponse,
    ExportData,
    ExportSummary,
    # Feedback types
    FeedbackResponse,
    FeedbackValue,
    # Project types
    MemberRole,
    # Memory types
    Memory,
    MemoryAddResponse,
    MemoryDeleteResponse,
    MemoryEvent,
    MemoryHistoryEntry,
    MemoryListResponse,
    MemoryResult,
    MemorySearchResponse,
    MemoryUpdateResponse,
    # Other
    PingResponse,
    Project,
    ProjectCreateResponse,
    ProjectMember,
    ProjectMembersResponse,
    # Webhook types
    Webhook,
    WebhookDeleteResponse,
    WebhookListResponse,
)

__all__ = [
    # Client
    "Mem0Client",
    # Convenience functions
    "add",
    "search",
    # Result types
    "Result",
    "Ok",
    "Err",
    # Error
    "ApiError",
    # Memory types
    "Memory",
    "MemoryAddResponse",
    "MemoryDeleteResponse",
    "MemoryEvent",
    "MemoryHistoryEntry",
    "MemoryListResponse",
    "MemoryResult",
    "MemorySearchResponse",
    "MemoryUpdateResponse",
    # Entity types
    "EntitiesResponse",
    "Entity",
    "EntityType",
    # Batch types
    "BatchDeleteResponse",
    "BatchMemoryDelete",
    "BatchMemoryUpdate",
    "BatchUpdateResponse",
    # Export types
    "ExportCreateResponse",
    "ExportData",
    "ExportSummary",
    # Webhook types
    "Webhook",
    "WebhookDeleteResponse",
    "WebhookListResponse",
    # Project types
    "MemberRole",
    "Project",
    "ProjectCreateResponse",
    "ProjectMember",
    "ProjectMembersResponse",
    # Feedback types
    "FeedbackResponse",
    "FeedbackValue",
    # Other
    "PingResponse",
]

__version__ = "1.0.0"

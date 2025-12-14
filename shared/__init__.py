"""
Shared utilities for Claude Skills.

This module provides common infrastructure for all skills in the Cognitive Construct.
Phase 1 Infrastructure complete (2025-12-13).
"""
from .credentials import (
    CredentialValidator,
    mask_credential,
    get_credential_inventory,
    CredentialRouter,
    CredentialStatus,
    get_credential_router,
)
from .errors import ErrorCode, SkillError, sanitize_error, format_error_response
from .response import UsageMetrics, TimingMetrics, SkillResponse
from .session import Session, SessionManager
from .validators import (
    SkillMessage,
    SkillMetadata,
    MessageType,
    SkillMessageBus,
    get_message_bus,
)
from .mcp_client import ServerConfig, MCPConnection, MCPClientPool, PoolConfig, get_client
from .synergies import (
    rhetoric_request_context,
    volition_log_action,
    encyclopedia_cache_result,
    register_all_synergies,
    unregister_all_synergies,
    get_synergy_status,
    ContextLookupResult,
    ActionLogEntry,
)
from .feature_flags import (
    FEATURE_FLAGS,
    get_flag,
    set_flag,
    get_all_flags,
    get_enabled_search_providers,
    get_enabled_llm_providers,
    is_synergy_enabled,
)
from .events import (
    Event,
    EventType,
    EventFilter,
    EventValidationError,
    StateSnapshot,
    create_event,
)
from .membrane import (
    Membrane,
    AbsorbRule,
    EmitSchema,
    Priority,
    RateLimits,
    get_membrane,
    register_membrane,
    SKILL_MEMBRANES,
)
from .feedback import (
    FeedbackSignal,
    FeedbackSummary,
    FeedbackCollector,
    SignalType,
    record_feedback,
    get_source_effectiveness,
    get_recommended_sources,
)

__all__ = [
    # Credentials
    "CredentialValidator",
    "mask_credential",
    "get_credential_inventory",
    "CredentialRouter",
    "CredentialStatus",
    "get_credential_router",
    # Errors
    "ErrorCode",
    "SkillError",
    "sanitize_error",
    "format_error_response",
    # Response
    "UsageMetrics",
    "TimingMetrics",
    "SkillResponse",
    # Session
    "Session",
    "SessionManager",
    # Validators / Message Schema
    "SkillMessage",
    "SkillMetadata",
    "MessageType",
    "SkillMessageBus",
    "get_message_bus",
    # MCP Client
    "ServerConfig",
    "MCPConnection",
    "MCPClientPool",
    "PoolConfig",
    "get_client",
    # Synergies
    "rhetoric_request_context",
    "volition_log_action",
    "encyclopedia_cache_result",
    "register_all_synergies",
    "unregister_all_synergies",
    "get_synergy_status",
    "ContextLookupResult",
    "ActionLogEntry",
    # Feature Flags
    "FEATURE_FLAGS",
    "get_flag",
    "set_flag",
    "get_all_flags",
    "get_enabled_search_providers",
    "get_enabled_llm_providers",
    "is_synergy_enabled",
    # Events (Phase 7)
    "Event",
    "EventType",
    "EventFilter",
    "EventValidationError",
    "StateSnapshot",
    "create_event",
    # Membranes (Phase 7)
    "Membrane",
    "AbsorbRule",
    "EmitSchema",
    "Priority",
    "RateLimits",
    "get_membrane",
    "register_membrane",
    "SKILL_MEMBRANES",
    # Feedback (Phase 7)
    "FeedbackSignal",
    "FeedbackSummary",
    "FeedbackCollector",
    "SignalType",
    "record_feedback",
    "get_source_effectiveness",
    "get_recommended_sources",
]

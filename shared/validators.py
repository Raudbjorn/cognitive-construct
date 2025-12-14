from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, Literal, List, Callable
from .errors import SkillError, ErrorCode, format_error_response
import datetime
import uuid
import asyncio
from enum import Enum


class MessageType(str, Enum):
    """Standard message types for inter-skill communication."""
    REQUEST = "request"           # Skill A requests something from Skill B
    RESPONSE = "response"         # Response to a request
    EVENT = "event"               # Notification event (fire-and-forget)
    CONTEXT_LOOKUP = "context_lookup"      # Rhetoric → Encyclopedia
    ACTION_LOG = "action_log"              # Volition → Inland Empire
    CACHE_STORE = "cache_store"            # Encyclopedia → Inland Empire


class SkillMessage(BaseModel):
    """
    Standard message format for inter-skill communication.
    Implements Requirement 8.4: Common JSON schema for inter-skill data sharing.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    source_skill: str = Field(..., description="Name of the sending skill")
    target_skill: str = Field(..., description="Name of the receiving skill")
    message_type: MessageType = Field(..., description="Type of message")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    payload: Dict[str, Any] = Field(default_factory=dict, description="The content of the message")
    session_id: Optional[str] = Field(None, description="Correlation ID for the session/task")
    correlation_id: Optional[str] = Field(None, description="ID linking request/response pairs")
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live for cached messages")

    @field_validator('source_skill', 'target_skill')
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        allowed_skills = {'encyclopedia', 'rhetoric', 'inland-empire', 'volition', 'system'}
        if v not in allowed_skills:
            raise ValueError(f"Unknown skill: {v}. Must be one of {allowed_skills}")
        return v

    @model_validator(mode='after')
    def validate_response_has_correlation(self) -> "SkillMessage":
        """Responses must have a correlation_id linking to the original request."""
        if self.message_type == MessageType.RESPONSE and not self.correlation_id:
            raise ValueError("Response messages must include correlation_id")
        return self

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "SkillMessage":
        try:
            return cls.model_validate_json(json_str)
        except Exception as e:
            raise SkillError(ErrorCode.USER_ERROR, f"Invalid message format: {str(e)}")

    def create_response(self, payload: Dict[str, Any], success: bool = True) -> "SkillMessage":
        """Create a response message to this request."""
        return SkillMessage(
            source_skill=self.target_skill,
            target_skill=self.source_skill,
            message_type=MessageType.RESPONSE,
            payload={"success": success, **payload},
            session_id=self.session_id,
            correlation_id=self.id
        )


class SkillMessageBus:
    """
    Simple in-process message bus for inter-skill communication.
    Implements Requirement 8.5: Synergies operate transparently without LLM orchestration.
    """
    _instance: Optional["SkillMessageBus"] = None
    _handlers: Dict[str, List[Callable[[SkillMessage], Optional[SkillMessage]]]]

    def __new__(cls) -> "SkillMessageBus":
        if cls._instance is None:
            cls._instance = super(SkillMessageBus, cls).__new__(cls)
            cls._instance._handlers = {}
            cls._instance._enabled = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register_handler(
        self,
        target_skill: str,
        handler: Callable[[SkillMessage], Optional[SkillMessage]]
    ) -> None:
        """Register a message handler for a skill."""
        if target_skill not in self._handlers:
            self._handlers[target_skill] = []
        self._handlers[target_skill].append(handler)

    def unregister_handlers(self, target_skill: str) -> None:
        """Remove all handlers for a skill."""
        if target_skill in self._handlers:
            del self._handlers[target_skill]

    def send(self, message: SkillMessage) -> Optional[SkillMessage]:
        """
        Send a message to target skill handlers.
        Returns the first response (if any) for request messages.
        """
        if not self._enabled:
            return None

        handlers = self._handlers.get(message.target_skill, [])
        for handler in handlers:
            try:
                response = handler(message)
                if response is not None and message.message_type == MessageType.REQUEST:
                    return response
            except Exception:
                # Synergies should fail silently (Requirement 8.5)
                pass
        return None

    async def send_async(self, message: SkillMessage) -> Optional[SkillMessage]:
        """Async version of send for handlers that need I/O."""
        if not self._enabled:
            return None

        handlers = self._handlers.get(message.target_skill, [])
        for handler in handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    response = await result
                else:
                    response = result
                if response is not None and message.message_type == MessageType.REQUEST:
                    return response
            except Exception:
                pass
        return None

    def enable(self) -> None:
        """Enable synergy messaging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable synergy messaging (for testing isolation)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Convenience function to get the singleton bus
def get_message_bus() -> SkillMessageBus:
    """Get the singleton message bus instance."""
    return SkillMessageBus()

class SkillMetadata(BaseModel):
    """
    Validates SKILL.md YAML frontmatter.
    """
    name: str = Field(..., max_length=64)
    description: str = Field(..., max_length=200)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dependencies: Optional[list[str]] = None

    @classmethod
    def parse_md(cls, content: str) -> "SkillMetadata":
        """
        Extracts and parses YAML frontmatter from markdown content.
        """
        import yaml
        if not content.startswith("---"):
            raise ValueError("Missing frontmatter start")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid frontmatter format")

        try:
            data = yaml.safe_load(parts[1])
            return cls(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        except Exception as e:
            raise ValueError(f"Validation error: {e}")

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
import time
from .errors import format_error_response

@dataclass
class UsageMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

@dataclass
class TimingMetrics:
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0

    def stop(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

@dataclass
class SkillResponse:
    content: Any # The main result
    success: bool = True
    error: Optional[Dict[str, Any]] = None
    messages: List[str] = field(default_factory=list) # User-facing status messages
    data: Optional[Dict[str, Any]] = None # Structured data
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)

    def __post_init__(self):
        if self.timing.end_time == 0.0:
            self.timing.stop()

    @classmethod
    def create_success(cls, content: Any, messages: List[str] = None, data: Dict[str, Any] = None):
        return cls(
            content=content,
            success=True,
            messages=messages or [],
            data=data
        )

    @classmethod
    def create_error(cls, error: Exception, context: str = None):
        err_dict = format_error_response(error)
        msg = f"Error: {err_dict['message']}"
        if context:
            msg = f"{context}: {msg}"

        return cls(
            content=None,
            success=False,
            error=err_dict,
            messages=[msg]
        )

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Any
import traceback
import sys

class ErrorCode(IntEnum):
    USER_ERROR = 1        # Invalid input, bad arguments
    SYSTEM_ERROR = 2      # Internal logic errors, file system issues
    UPSTREAM_ERROR = 3    # API failures, network timeouts
    SECURITY_ERROR = 4    # Permission denied, credential issues

@dataclass
class SkillError(Exception):
    code: ErrorCode
    message: str
    context: Optional[dict[str, Any]] = None
    original_error: Optional[Exception] = None

    def __str__(self):
        return f"Error {self.code}: {self.message}"

def sanitize_error(e: Exception) -> SkillError:
    """
    Converts any exception into a safe SkillError, hiding details for system/upstream errors
    unless debugging.
    """
    if isinstance(e, SkillError):
        return e

    # Map common exceptions
    if isinstance(e, (ValueError, TypeError)):
        return SkillError(ErrorCode.USER_ERROR, str(e), original_error=e)
    if isinstance(e, (PermissionError,)):
        return SkillError(ErrorCode.SECURITY_ERROR, "Permission denied", original_error=e)

    # Default to system error, but strip stack trace for user safety
    # In a real app we might log the full trace here
    return SkillError(
        ErrorCode.SYSTEM_ERROR,
        f"Internal error: {type(e).__name__}",
        original_error=e
    )

def format_error_response(e: Exception, include_traceback: bool = False) -> dict[str, Any]:
    """
    Formats an exception for the final response.
    """
    skill_error = sanitize_error(e)
    response = {
        "error": True,
        "code": skill_error.code.value,
        "message": skill_error.message
    }

    if skill_error.context:
        response["context"] = skill_error.context

    if include_traceback and skill_error.original_error:
        # Only recommended for internal logs, not user output usually
        # But for development it might be used via a flag
        response["traceback"] = "".join(traceback.format_exception(
            type(skill_error.original_error),
            skill_error.original_error,
            skill_error.original_error.__traceback__
        ))

    return response

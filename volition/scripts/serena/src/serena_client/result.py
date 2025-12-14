"""Result type for error handling as values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success case containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Get the value. Safe to call after checking is_ok()."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error case containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        """Raises ValueError. Check is_ok() first."""
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default since this is an error."""
        return default

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform success value (no-op for Err)."""
        return self  # type: ignore[return-value]

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error."""
        return Err(fn(self.error))


# Type alias for the union
Result = Union[Ok[T], Err[E]]

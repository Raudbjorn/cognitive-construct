"""Result type for error handling as values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success case containing a value."""

    value: T

    def is_ok(self) -> bool:
        """Check if this is a success result."""
        return True

    def is_err(self) -> bool:
        """Check if this is an error result."""
        return False

    def unwrap(self) -> T:
        """Get the value. Safe to call after checking is_ok()."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value

    def unwrap_or_else(self, fn: Callable[[Any], T]) -> T:
        """Get value or compute from error (no-op for Ok)."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore[return-value]

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that return Result."""
        return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Handle error case (no-op for Ok)."""
        return self  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error case containing an error."""

    error: E

    def is_ok(self) -> bool:
        """Check if this is a success result."""
        return False

    def is_err(self) -> bool:
        """Check if this is an error result."""
        return True

    def unwrap(self) -> Any:
        """Raises ValueError. Check is_ok() first."""
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default since this is an error."""
        return default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """Compute value from error."""
        return fn(self.error)

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """Transform success value (no-op for Err)."""
        return self  # type: ignore[return-value]

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """Transform the error."""
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations (no-op for Err)."""
        return self  # type: ignore[return-value]

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Handle error case."""
        return fn(self.error)

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# Type alias for the union
Result = Union[Ok[T], Err[E]]

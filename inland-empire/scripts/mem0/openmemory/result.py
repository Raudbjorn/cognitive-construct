"""Result type for error handling as values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success case containing a value."""

    value: T

    def is_ok(self) -> bool:
        """Return True if this is a success case."""
        return True

    def is_err(self) -> bool:
        """Return False since this is a success case."""
        return False

    def unwrap(self) -> T:
        """Get the value. Safe to call after checking is_ok()."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or return default. Returns value since this is Ok."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error (no-op for Ok)."""
        return self  # type: ignore


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error case containing an error."""

    error: E

    def is_ok(self) -> bool:
        """Return False since this is an error case."""
        return False

    def is_err(self) -> bool:
        """Return True if this is an error case."""
        return True

    def unwrap(self) -> T:
        """Raises ValueError. Check is_ok() first."""
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return the default since this is an error."""
        return default

    def map(self, fn: Callable[[T], U]) -> "Result[U, E]":
        """Transform success value (no-op for Err)."""
        return self  # type: ignore

    def map_err(self, fn: Callable[[E], U]) -> "Result[T, U]":
        """Transform the error."""
        return Err(fn(self.error))


Result = Union[Ok[T], Err[E]]

"""Result type for error handling as values.

Canonical implementation shared across all encyclopedia client libraries.
Replaces per-package duplicates in codegraph, context7client, exaclient,
gitingest, kagiclient, perplexity, and searxng.

Usage:
    from shared.result import Result, Ok, Err

    def divide(a: float, b: float) -> Result[float, str]:
        if b == 0:
            return Err("division by zero")
        return Ok(a / b)

    result = divide(10, 3)
    if result.is_ok():
        print(result.unwrap())  # 3.333...

    # Functional chaining
    doubled = divide(10, 3).map(lambda x: x * 2)
    assert doubled.unwrap() == 20 / 3

    # Error mapping
    tagged = divide(10, 0).map_err(lambda e: f"math: {e}")
    assert tagged.error == "math: division by zero"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

__all__ = ["Result", "Ok", "Err"]

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success case containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:  # type: ignore[override]
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], F]) -> Ok[T]:  # type: ignore[override]
        return self  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error case containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:  # type: ignore[type-var]
        raise ValueError(f"Called unwrap on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:  # type: ignore[type-var]
        return default

    def map(self, fn: Callable[[T], U]) -> Err[E]:  # type: ignore[type-var]
        return self  # type: ignore[return-value]

    def map_err(self, fn: Callable[[E], F]) -> Err[F]:
        return Err(fn(self.error))


Result = Union[Ok[T], Err[E]]

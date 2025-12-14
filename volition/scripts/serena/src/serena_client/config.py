"""Configuration types for the serena-client library."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self


class Language(str, Enum):
    """Supported programming languages for language server operations."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    CSHARP = "csharp"
    CPP = "cpp"
    RUBY = "ruby"
    DART = "dart"
    PHP = "php"
    PERL = "perl"
    CLOJURE = "clojure"
    ELIXIR = "elixir"
    ELM = "elm"
    TERRAFORM = "terraform"
    SWIFT = "swift"
    BASH = "bash"
    ZIG = "zig"
    LUA = "lua"
    NIX = "nix"
    ERLANG = "erlang"
    SCALA = "scala"
    JULIA = "julia"
    FORTRAN = "fortran"
    HASKELL = "haskell"
    VUE = "vue"
    POWERSHELL = "powershell"
    R = "r"
    AL = "al"
    FSHARP = "fsharp"
    REGO = "rego"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse a language from string, case-insensitive."""
        normalized = value.lower().strip()
        # Handle common aliases
        aliases = {
            "javascript": "typescript",
            "js": "typescript",
            "ts": "typescript",
            "py": "python",
            "rb": "ruby",
            "rs": "rust",
            "cs": "csharp",
            "c#": "csharp",
            "c++": "cpp",
            "f#": "fsharp",
        }
        normalized = aliases.get(normalized, normalized)
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Configuration for SerenaClient."""

    project_root: str
    languages: tuple[Language, ...] = ()
    timeout: float = 120.0
    encoding: str = "utf-8"
    ignored_paths: tuple[str, ...] = ()
    trace_lsp: bool = False

    @classmethod
    def create(
        cls,
        project_root: str,
        languages: list[Language] | None = None,
        timeout: float = 120.0,
        encoding: str = "utf-8",
        ignored_paths: list[str] | None = None,
        trace_lsp: bool = False,
    ) -> Self:
        """Create a ClientConfig with convenient list parameters."""
        return cls(
            project_root=project_root,
            languages=tuple(languages) if languages else (),
            timeout=timeout,
            encoding=encoding,
            ignored_paths=tuple(ignored_paths) if ignored_paths else (),
            trace_lsp=trace_lsp,
        )


# Default timeout for language server operations
DEFAULT_TIMEOUT: float = 120.0

# Default file encoding
DEFAULT_ENCODING: str = "utf-8"

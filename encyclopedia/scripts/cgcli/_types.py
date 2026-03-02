"""Type definitions for codegraph library."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    JAVA = "java"
    RUBY = "ruby"


EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".ipynb": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".c": Language.C,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cpp": Language.CPP,
    ".java": Language.JAVA,
    ".rb": Language.RUBY,
}


@dataclass(frozen=True, slots=True)
class FunctionInfo:
    """Information about a function."""

    name: str
    file_path: str
    line_number: int
    source: str | None = None
    docstring: str | None = None
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    is_dependency: bool = False
    cyclomatic_complexity: int | None = None


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Information about a class."""

    name: str
    file_path: str
    line_number: int
    source: str | None = None
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    is_dependency: bool = False


@dataclass(frozen=True, slots=True)
class VariableInfo:
    """Information about a variable."""

    name: str
    file_path: str
    line_number: int
    value: str | None = None
    context: str | None = None
    is_dependency: bool = False


@dataclass(frozen=True, slots=True)
class CallInfo:
    """Information about a function call relationship."""

    caller_name: str
    caller_file_path: str
    caller_line_number: int
    called_name: str
    called_file_path: str
    call_line_number: int
    args: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about an import."""

    file_name: str
    file_path: str
    module_name: str
    alias: str | None = None
    is_dependency: bool = False


@dataclass(frozen=True, slots=True)
class ClassHierarchy:
    """Class inheritance hierarchy."""

    class_name: str
    parent_classes: list[ClassInfo]
    child_classes: list[ClassInfo]
    methods: list[FunctionInfo]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A search result with relevance scoring."""

    name: str
    file_path: str
    line_number: int
    search_type: str
    relevance_score: float
    source: str | None = None
    docstring: str | None = None
    is_dependency: bool = False


@dataclass(frozen=True, slots=True)
class RelatedCodeResult:
    """Results from find_related_code."""

    query: str
    functions_by_name: list[FunctionInfo]
    classes_by_name: list[ClassInfo]
    variables_by_name: list[VariableInfo]
    content_matches: list[SearchResult]
    ranked_results: list[SearchResult]
    total_matches: int


@dataclass(frozen=True, slots=True)
class RepositoryInfo:
    """Information about an indexed repository."""

    name: str
    path: str
    is_dependency: bool


@dataclass(frozen=True, slots=True)
class CodeGraphError:
    """Error from code graph operations."""

    message: str
    details: str | None = None

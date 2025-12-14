"""Type definitions for the serena-client library.

All types are frozen dataclasses with slots for immutability and memory efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Self

# === Enums ===


class SymbolKind(IntEnum):
    """A symbol kind from the Language Server Protocol."""

    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


class CompletionItemKind(IntEnum):
    """The kind of a completion entry."""

    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25


class MarkupKind(str, Enum):
    """The type of markup content."""

    PlainText = "plaintext"
    Markdown = "markdown"


# === Position and Range ===


@dataclass(frozen=True, slots=True)
class Position:
    """Position in a text document (zero-based line and character)."""

    line: int
    character: int

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP Position dict."""
        return cls(
            line=data.get("line", 0),
            character=data.get("character", 0),
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to LSP-compatible dict."""
        return {"line": self.line, "character": self.character}


@dataclass(frozen=True, slots=True)
class Range:
    """A range in a text document."""

    start: Position
    end: Position

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP Range dict."""
        return cls(
            start=Position.from_lsp(data.get("start", {})),
            end=Position.from_lsp(data.get("end", {})),
        )

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Convert to LSP-compatible dict."""
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @property
    def line_count(self) -> int:
        """Number of lines spanned by this range."""
        return self.end.line - self.start.line + 1


@dataclass(frozen=True, slots=True)
class Location:
    """A location in a text document."""

    uri: str
    range: Range
    absolute_path: str
    relative_path: str | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP Location dict."""
        return cls(
            uri=data.get("uri", ""),
            range=Range.from_lsp(data.get("range", {})),
            absolute_path=data.get("absolutePath", ""),
            relative_path=data.get("relativePath"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to LSP-compatible dict."""
        result: dict[str, Any] = {
            "uri": self.uri,
            "range": self.range.to_dict(),
            "absolutePath": self.absolute_path,
        }
        if self.relative_path is not None:
            result["relativePath"] = self.relative_path
        return result


# === Symbol Types ===


@dataclass(frozen=True, slots=True)
class Symbol:
    """Represents a programming symbol (class, function, variable, etc.)."""

    name: str
    kind: SymbolKind
    location: Location | None = None
    range: Range | None = None
    selection_range: Range | None = None
    detail: str | None = None
    container_name: str | None = None
    deprecated: bool = False
    body: str | None = None
    children: tuple["Symbol", ...] = ()
    overload_idx: int | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP UnifiedSymbolInformation dict."""
        location = Location.from_lsp(data["location"]) if "location" in data else None
        range_ = Range.from_lsp(data["range"]) if "range" in data else None
        selection_range = Range.from_lsp(data["selectionRange"]) if "selectionRange" in data else None

        children_data = data.get("children", [])
        children = tuple(cls.from_lsp(c) for c in children_data) if children_data else ()

        return cls(
            name=data.get("name", ""),
            kind=SymbolKind(data.get("kind", SymbolKind.Variable)),
            location=location,
            range=range_,
            selection_range=selection_range,
            detail=data.get("detail"),
            container_name=data.get("containerName"),
            deprecated=data.get("deprecated", False),
            body=data.get("body"),
            children=children,
            overload_idx=data.get("overload_idx"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
        }
        if self.location is not None:
            result["location"] = self.location.to_dict()
        if self.range is not None:
            result["range"] = self.range.to_dict()
        if self.selection_range is not None:
            result["selectionRange"] = self.selection_range.to_dict()
        if self.detail is not None:
            result["detail"] = self.detail
        if self.container_name is not None:
            result["containerName"] = self.container_name
        if self.deprecated:
            result["deprecated"] = True
        if self.body is not None:
            result["body"] = self.body
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.overload_idx is not None:
            result["overload_idx"] = self.overload_idx
        return result

    @property
    def is_class(self) -> bool:
        return self.kind == SymbolKind.Class

    @property
    def is_function(self) -> bool:
        return self.kind in (SymbolKind.Function, SymbolKind.Method)

    @property
    def is_variable(self) -> bool:
        return self.kind in (SymbolKind.Variable, SymbolKind.Constant, SymbolKind.Field)


@dataclass(frozen=True, slots=True)
class DocumentSymbols:
    """Symbols for a document, organized hierarchically."""

    path: str
    symbols: tuple[Symbol, ...]

    @classmethod
    def from_lsp(cls, path: str, data: list[dict[str, Any]]) -> Self:
        """Create from list of LSP symbol dicts."""
        return cls(
            path=path,
            symbols=tuple(Symbol.from_lsp(s) for s in data),
        )

    @property
    def all_symbols(self) -> list[Symbol]:
        """Get all symbols flattened (including children)."""
        result: list[Symbol] = []

        def collect(symbols: tuple[Symbol, ...]) -> None:
            for s in symbols:
                result.append(s)
                if s.children:
                    collect(s.children)

        collect(self.symbols)
        return result


# === Completion Types ===


@dataclass(frozen=True, slots=True)
class CompletionItem:
    """A completion item for code completion."""

    label: str
    kind: CompletionItemKind
    detail: str | None = None
    insert_text: str | None = None
    documentation: str | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP CompletionItem dict."""
        # Handle both 'label' and 'completionText' (Serena's extension)
        label = data.get("label") or data.get("completionText", "")
        return cls(
            label=label,
            kind=CompletionItemKind(data.get("kind", CompletionItemKind.Text)),
            detail=data.get("detail"),
            insert_text=data.get("insertText"),
            documentation=data.get("documentation"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict representation."""
        result: dict[str, Any] = {
            "label": self.label,
            "kind": self.kind.value,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.insert_text is not None:
            result["insertText"] = self.insert_text
        if self.documentation is not None:
            result["documentation"] = self.documentation
        return result


# === Hover Types ===


@dataclass(frozen=True, slots=True)
class MarkupContent:
    """Markup content for hover and documentation."""

    kind: MarkupKind
    value: str

    @classmethod
    def from_lsp(cls, data: dict[str, Any] | str) -> Self:
        """Create from LSP MarkupContent or string."""
        if isinstance(data, str):
            return cls(kind=MarkupKind.PlainText, value=data)
        return cls(
            kind=MarkupKind(data.get("kind", "plaintext")),
            value=data.get("value", ""),
        )


@dataclass(frozen=True, slots=True)
class HoverResult:
    """Result of a hover request."""

    contents: MarkupContent
    range: Range | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP Hover dict."""
        contents_data = data.get("contents", {})
        # Handle various content formats
        if isinstance(contents_data, str):
            contents = MarkupContent(kind=MarkupKind.PlainText, value=contents_data)
        elif isinstance(contents_data, list):
            # Concatenate multiple items
            parts = []
            for item in contents_data:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "value" in item:
                    parts.append(item["value"])
            contents = MarkupContent(kind=MarkupKind.Markdown, value="\n\n".join(parts))
        else:
            contents = MarkupContent.from_lsp(contents_data)

        range_ = Range.from_lsp(data["range"]) if "range" in data else None
        return cls(contents=contents, range=range_)


# === Edit Types ===


@dataclass(frozen=True, slots=True)
class TextEdit:
    """A textual edit to a document."""

    range: Range
    new_text: str

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP TextEdit dict."""
        return cls(
            range=Range.from_lsp(data.get("range", {})),
            new_text=data.get("newText", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to LSP-compatible dict."""
        return {
            "range": self.range.to_dict(),
            "newText": self.new_text,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceEdit:
    """A workspace edit representing changes to multiple documents."""

    changes: dict[str, tuple[TextEdit, ...]] = field(default_factory=dict)

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP WorkspaceEdit dict."""
        changes: dict[str, tuple[TextEdit, ...]] = {}

        # Handle 'changes' format
        if "changes" in data:
            for uri, edits in data["changes"].items():
                changes[uri] = tuple(TextEdit.from_lsp(e) for e in edits)

        # Handle 'documentChanges' format
        elif "documentChanges" in data:
            for change in data["documentChanges"]:
                if "textDocument" in change and "edits" in change:
                    uri = change["textDocument"]["uri"]
                    changes[uri] = tuple(TextEdit.from_lsp(e) for e in change["edits"])

        return cls(changes=changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to LSP-compatible dict."""
        return {
            "changes": {uri: [e.to_dict() for e in edits] for uri, edits in self.changes.items()},
        }

    @property
    def is_empty(self) -> bool:
        """Check if this edit contains no changes."""
        return len(self.changes) == 0

    @property
    def affected_files(self) -> list[str]:
        """Get list of file URIs affected by this edit."""
        return list(self.changes.keys())


# === Diagnostic Types ===


class DiagnosticSeverity(IntEnum):
    """Severity of a diagnostic."""

    Error = 1
    Warning = 2
    Information = 3
    Hint = 4


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A diagnostic (error, warning, etc.) for a document."""

    uri: str
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.Error
    code: str | None = None
    source: str | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Self:
        """Create from LSP Diagnostic dict."""
        severity_val = data.get("severity", 1)
        return cls(
            uri=data.get("uri", ""),
            range=Range.from_lsp(data.get("range", {})),
            message=data.get("message", ""),
            severity=DiagnosticSeverity(severity_val),
            code=data.get("code"),
            source=data.get("source"),
        )

    @property
    def is_error(self) -> bool:
        return self.severity == DiagnosticSeverity.Error

    @property
    def is_warning(self) -> bool:
        return self.severity == DiagnosticSeverity.Warning

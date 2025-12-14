"""
serena-client - Pure Python client library for language server operations

Provides programmatic access to language server capabilities including symbol
navigation, code completion, hover information, and refactoring operations.
All operations return Result types for explicit error handling.

Usage:
    from serena_client import SerenaClient, Language

    # Create client for a project
    client = SerenaClient(
        project_root="/path/to/project",
        languages=[Language.PYTHON],
    )

    # Use as context manager for automatic cleanup
    with client.start():
        # Find symbols in workspace
        result = client.find_symbols("MyClass")
        if result.is_ok():
            for symbol in result.value:
                print(f"{symbol.name}: {symbol.kind}")
        else:
            print(f"Error: {result.error.message}")

        # Get document symbols
        result = client.get_document_symbols("src/main.py")
        if result.is_ok():
            for symbol in result.value.symbols:
                print(f"  {symbol.name}")

        # Find definition
        result = client.find_definition("src/main.py", line=10, column=5)

        # Get hover info
        result = client.get_hover("src/main.py", line=10, column=5)

    # Convenience functions for one-off operations
    from serena_client import get_document_symbols, find_symbols

    result = get_document_symbols("/path/to/project", "src/main.py", Language.PYTHON)
"""

from .client import SerenaClient, find_symbols, get_document_symbols
from .config import ClientConfig, Language
from .errors import (
    ClientError,
    ConfigurationError,
    FileNotFoundError,
    LanguageServerError,
    SymbolNotFoundError,
)
from .result import Err, Ok, Result
from .types import (
    CompletionItem,
    CompletionItemKind,
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbols,
    HoverResult,
    Location,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
    Symbol,
    SymbolKind,
    TextEdit,
    WorkspaceEdit,
)

__all__ = [
    "ClientConfig",
    "ClientError",
    "CompletionItem",
    "CompletionItemKind",
    "ConfigurationError",
    "Diagnostic",
    "DiagnosticSeverity",
    "DocumentSymbols",
    "Err",
    "FileNotFoundError",
    "HoverResult",
    "Language",
    "LanguageServerError",
    "Location",
    "MarkupContent",
    "MarkupKind",
    "Ok",
    "Position",
    "Range",
    "Result",
    "SerenaClient",
    "Symbol",
    "SymbolKind",
    "SymbolNotFoundError",
    "TextEdit",
    "WorkspaceEdit",
    "find_symbols",
    "get_document_symbols",
]

__version__ = "0.1.0"

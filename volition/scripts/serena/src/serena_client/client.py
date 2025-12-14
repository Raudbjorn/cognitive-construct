"""SerenaClient - Pure Python client for language server operations.

Usage:
    from serena_client import SerenaClient, Language

    client = SerenaClient(
        project_root="/path/to/project",
        languages=[Language.PYTHON],
    )

    with client.start():
        result = client.find_symbols("MyClass")
        if result.is_ok():
            for symbol in result.value:
                print(f"{symbol.name}: {symbol.kind}")
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from .config import DEFAULT_ENCODING, DEFAULT_TIMEOUT, Language
from .errors import ClientError, ConfigurationError, LanguageServerError
from .result import Err, Ok, Result
from .types import (
    CompletionItem,
    DocumentSymbols,
    HoverResult,
    Location,
    Symbol,
    WorkspaceEdit,
)

if TYPE_CHECKING:
    from solidlsp import SolidLanguageServer
    from solidlsp.ls import DocumentSymbols as LSPDocumentSymbols


@dataclass
class SerenaClient:
    """Client for language server operations on a project.

    Provides Result-based error handling for all operations.
    Use as a context manager for automatic resource cleanup.

    Example:
        client = SerenaClient(
            project_root="/path/to/project",
            languages=[Language.PYTHON],
        )
        with client.start():
            result = client.get_document_symbols("src/main.py")
            if result.is_ok():
                for symbol in result.value.symbols:
                    print(symbol.name)

    """

    project_root: str
    languages: list[Language] = field(default_factory=list)
    timeout: float = DEFAULT_TIMEOUT
    encoding: str = DEFAULT_ENCODING
    ignored_paths: list[str] = field(default_factory=list)
    trace_lsp: bool = False

    # Internal state
    _ls_manager: Any = field(init=False, repr=False, default=None)
    _started: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.project_root:
            raise ValueError("project_root is required")
        self.project_root = os.path.abspath(self.project_root)

    # === Lifecycle ===

    @contextmanager
    def start(self) -> Iterator[SerenaClient]:
        """Start language servers and yield self for use in with-block.

        Example:
            with client.start():
                result = client.find_symbols("MyClass")

        """
        self._start_servers()
        try:
            yield self
        finally:
            self.stop()

    def _start_servers(self) -> None:
        """Start language servers for configured languages."""
        if self._started:
            return

        if not self.languages:
            raise ValueError("At least one language must be specified")

        from solidlsp.ls_config import Language as LSPLanguage

        from .ls_manager import LanguageServerFactory, LanguageServerManager

        # Convert our Language enum to solidlsp's Language enum
        lsp_languages = [LSPLanguage(lang.value) for lang in self.languages]

        factory = LanguageServerFactory(
            project_root=self.project_root,
            encoding=self.encoding,
            ignored_patterns=self.ignored_paths,
            ls_timeout=self.timeout,
            trace_lsp_communication=self.trace_lsp,
        )

        self._ls_manager = LanguageServerManager.from_languages(lsp_languages, factory)
        self._started = True

    def stop(self) -> None:
        """Stop all language servers and release resources."""
        if self._ls_manager is not None:
            try:
                self._ls_manager.stop_all()
            except Exception:
                pass  # Ignore errors during cleanup
            self._ls_manager = None
        self._started = False

    def _check_started(self) -> Result[None, ClientError]:
        """Verify client is started."""
        if not self._started or self._ls_manager is None:
            return Err(
                ConfigurationError(
                    message="Client not started. Use 'with client.start():' context manager.",
                    code="NOT_STARTED",
                )
            )
        return Ok(None)

    def _get_ls_for_file(self, relative_path: str) -> Result["SolidLanguageServer", ClientError]:
        """Get the appropriate language server for a file."""
        check = self._check_started()
        if check.is_err():
            return check  # type: ignore[return-value]

        try:
            ls = self._ls_manager.get_language_server_for_path(relative_path)
            if ls is None:
                return Err(
                    ClientError(
                        message=f"No language server available for file: {relative_path}",
                        code="NO_LS_FOR_FILE",
                    )
                )
            return Ok(ls)
        except Exception as e:
            return Err(ClientError(message=str(e), code="LS_ERROR"))

    # === Symbol Operations ===

    def get_document_symbols(self, relative_path: str) -> Result[DocumentSymbols, ClientError]:
        """Get all symbols in a document.

        Args:
            relative_path: Path relative to project root

        Returns:
            Result containing DocumentSymbols on success or ClientError on failure

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            doc_symbols: LSPDocumentSymbols = ls.request_document_symbols(relative_path)
            symbols = tuple(Symbol.from_lsp(cast(dict[str, Any], s)) for s in doc_symbols.root_symbols)
            return Ok(DocumentSymbols(path=relative_path, symbols=symbols))
        except Exception as e:
            return Err(
                LanguageServerError(
                    message=str(e),
                    code="DOCUMENT_SYMBOLS_ERROR",
                    language=ls.language_id,
                )
            )

    def find_symbols(
        self,
        pattern: str,
        within_path: str | None = None,
    ) -> Result[list[Symbol], ClientError]:
        """Find symbols matching a pattern in the workspace.

        Args:
            pattern: Symbol name pattern to search for
            within_path: Optional path to limit search scope

        Returns:
            Result containing list of matching symbols or ClientError

        """
        check = self._check_started()
        if check.is_err():
            return check  # type: ignore[return-value]

        try:
            # Use the default language server for workspace queries
            ls = self._ls_manager.get_default_language_server()
            raw_symbols = ls.request_workspace_symbol(pattern)
            if raw_symbols is None:
                return Ok([])

            symbols = [Symbol.from_lsp(cast(dict[str, Any], s)) for s in raw_symbols]

            # Filter by path if specified
            if within_path:
                symbols = [
                    s for s in symbols if s.location and s.location.relative_path and s.location.relative_path.startswith(within_path)
                ]

            return Ok(symbols)
        except Exception as e:
            return Err(
                LanguageServerError(
                    message=str(e),
                    code="WORKSPACE_SYMBOL_ERROR",
                )
            )

    def get_symbol_body(self, relative_path: str, symbol_name: str) -> Result[str, ClientError]:
        """Get the source code body of a symbol.

        Args:
            relative_path: Path to the file containing the symbol
            symbol_name: Name of the symbol to retrieve

        Returns:
            Result containing the symbol's source code or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            doc_symbols = ls.request_document_symbols(relative_path)
            for s in doc_symbols.iter_symbols():
                if s.get("name") == symbol_name:
                    body = s.get("body")
                    if body:
                        return Ok(body)
                    return Err(
                        ClientError(
                            message=f"Symbol '{symbol_name}' has no body",
                            code="NO_SYMBOL_BODY",
                        )
                    )
            return Err(
                ClientError(
                    message=f"Symbol '{symbol_name}' not found in {relative_path}",
                    code="SYMBOL_NOT_FOUND",
                )
            )
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))

    # === Navigation ===

    def find_definition(
        self,
        relative_path: str,
        line: int,
        column: int,
    ) -> Result[list[Location], ClientError]:
        """Find the definition of a symbol at a position.

        Args:
            relative_path: Path relative to project root
            line: Zero-based line number
            column: Zero-based column number

        Returns:
            Result containing list of definition locations or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            locations = ls.request_definition(relative_path, line, column)
            return Ok([Location.from_lsp(cast(dict[str, Any], loc)) for loc in locations])
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))

    def find_references(
        self,
        relative_path: str,
        line: int,
        column: int,
    ) -> Result[list[Location], ClientError]:
        """Find all references to a symbol at a position.

        Args:
            relative_path: Path relative to project root
            line: Zero-based line number
            column: Zero-based column number

        Returns:
            Result containing list of reference locations or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            locations = ls.request_references(relative_path, line, column)
            return Ok([Location.from_lsp(cast(dict[str, Any], loc)) for loc in locations])
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))

    # === Code Intelligence ===

    def get_completions(
        self,
        relative_path: str,
        line: int,
        column: int,
    ) -> Result[list[CompletionItem], ClientError]:
        """Get code completions at a position.

        Args:
            relative_path: Path relative to project root
            line: Zero-based line number
            column: Zero-based column number

        Returns:
            Result containing list of completion items or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            completions = ls.request_completions(relative_path, line, column)
            return Ok([CompletionItem.from_lsp(cast(dict[str, Any], c)) for c in completions])
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))

    def get_hover(
        self,
        relative_path: str,
        line: int,
        column: int,
    ) -> Result[HoverResult | None, ClientError]:
        """Get hover information at a position.

        Args:
            relative_path: Path relative to project root
            line: Zero-based line number
            column: Zero-based column number

        Returns:
            Result containing HoverResult or None if no hover info, or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            hover = ls.request_hover(relative_path, line, column)
            if hover is None:
                return Ok(None)
            return Ok(HoverResult.from_lsp(cast(dict[str, Any], hover)))
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))

    # === Refactoring ===

    def prepare_rename(
        self,
        relative_path: str,
        line: int,
        column: int,
        new_name: str,
    ) -> Result[WorkspaceEdit, ClientError]:
        """Prepare a rename refactoring (does NOT apply changes).

        Args:
            relative_path: Path relative to project root
            line: Zero-based line number
            column: Zero-based column number
            new_name: New name for the symbol

        Returns:
            Result containing WorkspaceEdit describing the changes or ClientError

        """
        ls_result = self._get_ls_for_file(relative_path)
        if isinstance(ls_result, Err):
            return ls_result

        ls = ls_result.value
        try:
            edit = ls.request_rename_symbol_edit(relative_path, line, column, new_name)
            if edit is None:
                return Err(
                    ClientError(
                        message="Rename not supported at this location",
                        code="RENAME_NOT_SUPPORTED",
                    )
                )
            return Ok(WorkspaceEdit.from_lsp(cast(dict[str, Any], edit)))
        except Exception as e:
            return Err(LanguageServerError(message=str(e), language=ls.language_id))


# === Convenience Functions ===


def get_document_symbols(
    project_root: str,
    relative_path: str,
    language: Language,
) -> Result[DocumentSymbols, ClientError]:
    """Get document symbols (convenience function for one-off use).

    Creates a client for single use. For multiple requests,
    prefer creating a SerenaClient instance.
    """
    client = SerenaClient(project_root=project_root, languages=[language])
    with client.start():
        return client.get_document_symbols(relative_path)


def find_symbols(
    project_root: str,
    pattern: str,
    language: Language,
) -> Result[list[Symbol], ClientError]:
    """Find symbols matching a pattern (convenience function for one-off use).

    Creates a client for single use. For multiple requests,
    prefer creating a SerenaClient instance.
    """
    client = SerenaClient(project_root=project_root, languages=[language])
    with client.start():
        return client.find_symbols(pattern)

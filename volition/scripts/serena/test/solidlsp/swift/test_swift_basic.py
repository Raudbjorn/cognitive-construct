"""
Basic integration tests for the Swift language server functionality.

These tests validate the functionality of the language server APIs
like request_references using the Swift test repository.
"""

import os
import platform

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language

WINDOWS_SKIP = platform.system() == "Windows"
WINDOWS_SKIP_REASON = "GitHub Actions configuration for Swift on Windows is complex, skipping for now."

pytestmark = [pytest.mark.swift, pytest.mark.skipif(WINDOWS_SKIP, reason=WINDOWS_SKIP_REASON)]


class TestSwiftLanguageServerBasics:
    """Test basic functionality of the Swift language server."""

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_goto_definition_calculator_class(self, language_server: SolidLanguageServer) -> None:
        """Test goto_definition on Calculator class usage."""
        file_path = os.path.join("src", "main.swift")

        definitions = language_server.request_definition(file_path, 4, 23)
        assert isinstance(definitions, list), "Definitions should be a list"
        assert len(definitions) > 0, "Should find definition for Calculator class"

        calculator_def = definitions[0]
        assert calculator_def.get("uri", "").endswith("main.swift")

        start_line = calculator_def.get("range", {}).get("start", {}).get("line")
        assert start_line == 15, f"Calculator class definition should be at line 16, got {start_line + 1}"

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_goto_definition_user_struct(self, language_server: SolidLanguageServer) -> None:
        """Test goto_definition on User struct usage."""
        file_path = os.path.join("src", "main.swift")

        definitions = language_server.request_definition(file_path, 8, 18)
        assert isinstance(definitions, list)
        assert len(definitions) > 0

        user_def = definitions[0]
        assert user_def.get("uri", "").endswith("main.swift")

        start_line = user_def.get("range", {}).get("start", {}).get("line")
        assert start_line == 25, f"User struct definition should be at line 26, got {start_line + 1}"

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_goto_definition_calculator_method(self, language_server: SolidLanguageServer) -> None:
        """Test goto_definition on Calculator method usage."""
        file_path = os.path.join("src", "main.swift")

        definitions = language_server.request_definition(file_path, 5, 28)
        assert isinstance(definitions, list)

        add_def = definitions[0]
        assert add_def.get("uri", "").endswith("main.swift")

        start_line = add_def.get("range", {}).get("start", {}).get("line")
        assert start_line == 16, f"add method definition should be at line 17, got {start_line + 1}"

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_goto_definition_cross_file(self, language_server: SolidLanguageServer) -> None:
        """Test goto_definition across files - Utils struct."""
        utils_file = os.path.join("src", "utils.swift")

        symbols = language_server.request_document_symbols(utils_file).get_all_symbols_and_roots()
        utils_symbol = next((s for s in symbols[0] if s.get("name") == "Utils"), None)

        sel_start = utils_symbol["selectionRange"]["start"]
        definitions = language_server.request_definition(utils_file, sel_start["line"], sel_start["character"])
        assert isinstance(definitions, list)

        utils_def = definitions[0]
        assert utils_def.get("uri", "").endswith("utils.swift")

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_request_references_calculator_class(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on the Calculator class."""
        file_path = os.path.join("src", "main.swift")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        calculator_symbol = next((s for s in symbols[0] if s.get("name") == "Calculator"), None)

        sel_start = calculator_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert isinstance(references, list)
        assert len(references) > 0

        calculator_refs = [ref for ref in references if ref.get("uri", "").endswith("main.swift")]
        assert len(calculator_refs) > 0

        line_5_refs = [ref for ref in calculator_refs if ref.get("range", {}).get("start", {}).get("line") == 4]
        assert len(line_5_refs) > 0

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_request_references_user_struct(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on the User struct."""
        file_path = os.path.join("src", "main.swift")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        user_symbol = next((s for s in symbols[0] if s.get("name") == "User"), None)

        sel_start = user_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert isinstance(references, list)

        user_refs = [ref for ref in references if ref.get("uri", "").endswith("main.swift")]
        assert len(user_refs) > 0

        line_9_refs = [ref for ref in user_refs if ref.get("range", {}).get("start", {}).get("line") == 8]
        assert len(line_9_refs) > 0

    @pytest.mark.parametrize("language_server", [Language.SWIFT], indirect=True)
    def test_request_references_utils_struct(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on the Utils struct."""
        file_path = os.path.join("src", "utils.swift")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()
        utils_symbol = next((s for s in symbols[0] if s.get("name") == "Utils"), None)
        if not utils_symbol or "selectionRange" not in utils_symbol:
            raise AssertionError("Utils symbol or its selectionRange not found")
        sel_start = utils_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert isinstance(references, list)
        assert len(references) > 0

        utils_refs = [ref for ref in references if ref.get("uri", "").endswith("main.swift")]
        assert len(utils_refs) > 0

        line_12_refs = [ref for ref in utils_refs if ref.get("range", {}).get("start", {}).get("line") == 11]
        assert len(line_12_refs) > 0

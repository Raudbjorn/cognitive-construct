"""
Basic integration tests for the language server functionality.

These tests validate the functionality of the language server APIs
like request_references using the test repository.
"""

import os

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language


@pytest.mark.python
class TestLanguageServerBasics:
    """Test basic functionality of the language server."""

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_references_user_class(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on the User class."""
        file_path = os.path.join("test_repo", "models.py")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()
        user_symbol = next((s for s in symbols[0] if s.get("name") == "User"), None)
        if not user_symbol or "selectionRange" not in user_symbol:
            raise AssertionError("User symbol or its selectionRange not found")
        sel_start = user_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert len(references) > 1, "User class should be referenced in multiple files"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_references_item_class(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on the Item class."""
        file_path = os.path.join("test_repo", "models.py")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()
        item_symbol = next((s for s in symbols[0] if s.get("name") == "Item"), None)
        if not item_symbol or "selectionRange" not in item_symbol:
            raise AssertionError("Item symbol or its selectionRange not found")
        sel_start = item_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        services_references = [ref for ref in references if "services.py" in ref["uri"]]
        assert len(services_references) > 0, "At least one reference should be in services.py"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_references_function_parameter(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on a function parameter."""
        file_path = os.path.join("test_repo", "services.py")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()
        get_user_symbol = next((s for s in symbols[0] if s.get("name") == "get_user"), None)
        if not get_user_symbol or "selectionRange" not in get_user_symbol:
            raise AssertionError("get_user symbol or its selectionRange not found")
        sel_start = get_user_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert len(references) > 0, "id parameter should be referenced within the method"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_references_create_user_method(self, language_server: SolidLanguageServer) -> None:
        """Test request_references on create_user method."""
        file_path = os.path.join("test_repo", "services.py")
        symbols = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()
        create_user_symbol = next((s for s in symbols[0] if s.get("name") == "create_user"), None)
        if not create_user_symbol or "selectionRange" not in create_user_symbol:
            raise AssertionError("create_user symbol or its selectionRange not found")
        sel_start = create_user_symbol["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert len(references) > 1, "Should get valid references for create_user"

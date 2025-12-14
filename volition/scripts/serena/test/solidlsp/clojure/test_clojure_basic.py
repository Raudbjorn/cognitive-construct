import pytest

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls_types import UnifiedSymbolInformation
from test.conftest import language_tests_enabled

from . import CORE_PATH, UTILS_PATH


@pytest.mark.skipif(not language_tests_enabled(Language.CLOJURE), reason="Clojure tests are disabled")
@pytest.mark.clojure
class TestLanguageServerBasics:
    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_basic_definition(self, language_server: SolidLanguageServer):
        """Test finding definition of 'greet' function call in core.clj"""
        result = language_server.request_definition(CORE_PATH, 20, 12)

        assert isinstance(result, list)
        assert len(result) >= 1

        definition = result[0]
        assert definition["relativePath"] == CORE_PATH
        assert definition["range"]["start"]["line"] == 2, "Should find the definition of greet function at line 2"

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_cross_file_references(self, language_server: SolidLanguageServer):
        """Test finding references to 'multiply' function from core.clj"""
        result = language_server.request_references(CORE_PATH, 12, 6)

        assert isinstance(result, list) and len(result) >= 2, "Should find definition + usage in utils.clj"

        usage_found = any(item["relativePath"] == UTILS_PATH and item["range"]["start"]["line"] == 6 for item in result)
        assert usage_found, "Should find multiply usage in utils.clj"

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_completions(self, language_server: SolidLanguageServer):
        with language_server.open_file(UTILS_PATH):
            result = language_server.request_completions(UTILS_PATH, 6, 8)

            assert isinstance(result, list) and len(result) > 0

            completion_texts = [item["completionText"] for item in result]
            assert any("multiply" in text for text in completion_texts)

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_document_symbols(self, language_server: SolidLanguageServer):
        symbols, _ = language_server.request_document_symbols(CORE_PATH).get_all_symbols_and_roots()

        assert isinstance(symbols, list) and len(symbols) >= 4

        symbol_names = [symbol["name"] for symbol in symbols]
        expected_functions = ["greet", "add", "multiply", "-main"]

        for func_name in expected_functions:
            assert func_name in symbol_names, f"Should find {func_name} function in symbols"

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_hover(self, language_server: SolidLanguageServer):
        """Test hover on greet function"""
        result = language_server.request_hover(CORE_PATH, 2, 7)

        assert result is not None, "Hover should return information for greet function"
        assert "contents" in result
        contents = result["contents"]
        if isinstance(contents, str):
            assert "greet" in contents.lower()
        elif isinstance(contents, dict) and "value" in contents:
            assert "greet" in contents["value"].lower()
        else:
            assert False, f"Unexpected contents format: {type(contents)}"

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_workspace_symbols(self, language_server: SolidLanguageServer):
        result = language_server.request_workspace_symbol("add")

        assert isinstance(result, list) and len(result) > 0

        symbol_names = [symbol["name"] for symbol in result]
        assert any("add" in name.lower() for name in symbol_names)

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_namespace_functions(self, language_server: SolidLanguageServer):
        """Test definition lookup for core/greet usage in utils.clj"""
        result = language_server.request_definition(UTILS_PATH, 11, 25)

        assert isinstance(result, list)
        assert len(result) >= 1

        definition = result[0]
        assert definition["relativePath"] == CORE_PATH

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_request_references_with_content(self, language_server: SolidLanguageServer):
        """Test references to multiply function with content"""
        references = language_server.request_references(CORE_PATH, 12, 6)
        result = [
            language_server.retrieve_content_around_line(ref1["relativePath"], ref1["range"]["start"]["line"], 3, 0) for ref1 in references
        ]

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 2

        for ref in result:
            assert ref.source_file_path is not None
            content_str = ref.to_display_string()
            assert len(content_str) > 0

        utils_refs = [ref for ref in result if ref.source_file_path and "utils.clj" in ref.source_file_path]
        assert len(utils_refs) > 0

        utils_content = utils_refs[0].to_display_string()
        assert "calculate-area" in utils_content

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_request_full_symbol_tree(self, language_server: SolidLanguageServer):
        """Test retrieving the full symbol tree for project overview"""
        result = language_server.request_full_symbol_tree()

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        def list_all_symbols(symbols: list[UnifiedSymbolInformation]):
            found = []
            for symbol in symbols:
                found.append(symbol["name"])
                found.extend(list_all_symbols(symbol["children"]))
            return found

        all_symbol_names = list_all_symbols(result)

        expected_symbols = ["greet", "add", "multiply", "-main", "calculate-area", "format-greeting", "sum-list"]
        found_expected = [name for name in expected_symbols if any(name in symbol_name for symbol_name in all_symbol_names)]

        if len(found_expected) < 7:
            pytest.fail(f"Expected to find at least 3 symbols from {expected_symbols}, but found: {found_expected}")

    @pytest.mark.parametrize("language_server", [Language.CLOJURE], indirect=True)
    def test_request_referencing_symbols(self, language_server: SolidLanguageServer):
        """Test finding symbols that reference a given symbol"""
        result = language_server.request_referencing_symbols(CORE_PATH, 12, 6)
        assert isinstance(result, list) and len(result) > 0

        found_relevant_references = False
        for ref in result:
            if hasattr(ref, "symbol") and "calculate-area" in ref.symbol["name"]:
                found_relevant_references = True
                break

        assert found_relevant_references

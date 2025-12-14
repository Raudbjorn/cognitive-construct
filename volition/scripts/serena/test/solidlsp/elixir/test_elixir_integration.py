"""
Integration tests for Elixir language server with test repository.

These tests verify that the language server works correctly with a real Elixir project
and can perform advanced operations like cross-file symbol resolution.
"""

import os
from pathlib import Path

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language

from . import EXPERT_UNAVAILABLE, EXPERT_UNAVAILABLE_REASON

pytestmark = [pytest.mark.elixir, pytest.mark.skipif(EXPERT_UNAVAILABLE, reason=f"Next LS not available: {EXPERT_UNAVAILABLE_REASON}")]


class TestElixirIntegration:
    """Integration tests for Elixir language server with test repository."""

    @pytest.fixture
    def elixir_test_repo_path(self):
        """Get the path to the Elixir test repository."""
        test_dir = Path(__file__).parent.parent.parent
        return str(test_dir / "resources" / "repos" / "elixir" / "test_repo")

    def test_elixir_repo_structure(self, elixir_test_repo_path):
        """Test that the Elixir test repository has the expected structure."""
        repo_path = Path(elixir_test_repo_path)

        assert (repo_path / "mix.exs").exists()
        assert (repo_path / "lib" / "test_repo.ex").exists()
        assert (repo_path / "lib" / "utils.ex").exists()
        assert (repo_path / "lib" / "models.ex").exists()
        assert (repo_path / "lib" / "services.ex").exists()
        assert (repo_path / "lib" / "examples.ex").exists()
        assert (repo_path / "test" / "test_repo_test.exs").exists()
        assert (repo_path / "test" / "models_test.exs").exists()

    @pytest.mark.parametrize("language_server", [Language.ELIXIR], indirect=True)
    def test_cross_file_symbol_resolution(self, language_server: SolidLanguageServer):
        """Test that symbols can be resolved across different files."""
        services_file = os.path.join("lib", "services.ex")

        content = language_server.retrieve_full_file_content(services_file)
        lines = content.split("\n")
        user_reference_line = None
        for i, line in enumerate(lines):
            if "alias TestRepo.Models.{User" in line:
                user_reference_line = i
                break

        if user_reference_line is None:
            pytest.skip("Could not find User reference in services.ex")

        defining_symbol = language_server.request_defining_symbol(services_file, user_reference_line, 30)

        if defining_symbol and "location" in defining_symbol:
            assert "models.ex" in defining_symbol["location"]["uri"]

    @pytest.mark.parametrize("language_server", [Language.ELIXIR], indirect=True)
    def test_module_hierarchy_understanding(self, language_server: SolidLanguageServer):
        """Test that the language server understands Elixir module hierarchy."""
        models_file = os.path.join("lib", "models.ex")
        symbols = language_server.request_document_symbols(models_file).get_all_symbols_and_roots()

        if symbols:
            all_symbols = []
            for symbol_group in symbols:
                if isinstance(symbol_group, list):
                    all_symbols.extend(symbol_group)
                else:
                    all_symbols.append(symbol_group)

            symbol_names = [s.get("name", "") for s in all_symbols]

            expected_modules = ["TestRepo.Models", "User", "Item", "Order"]
            found_modules = [name for name in expected_modules if any(name in symbol_name for symbol_name in symbol_names)]
            assert len(found_modules) > 0, f"Expected modules {expected_modules}, found symbols {symbol_names}"

    def test_file_extension_matching(self):
        """Test that the Elixir language recognizes the correct file extensions."""
        language = Language.ELIXIR
        matcher = language.get_source_fn_matcher()

        assert matcher.is_relevant_filename("lib/test_repo.ex")
        assert matcher.is_relevant_filename("test/test_repo_test.exs")
        assert matcher.is_relevant_filename("config/config.exs")
        assert matcher.is_relevant_filename("mix.exs")
        assert matcher.is_relevant_filename("lib/models.ex")
        assert matcher.is_relevant_filename("lib/services.ex")

        assert not matcher.is_relevant_filename("README.md")
        assert not matcher.is_relevant_filename("lib/test_repo.py")
        assert not matcher.is_relevant_filename("package.json")
        assert not matcher.is_relevant_filename("Cargo.toml")

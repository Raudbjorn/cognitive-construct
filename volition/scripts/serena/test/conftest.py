"""Test configuration and fixtures for solidlsp tests."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.settings import SolidLSPSettings

from .solidlsp.clojure import is_clojure_cli_available

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Default directory name for project-local data
DEFAULT_PROJECT_DATA_DIR = ".serena"


def _get_user_data_dir() -> str:
    """Get the user data directory for language server caches."""
    home_dir = os.getenv("SERENA_HOME")
    if home_dir and home_dir.strip():
        return home_dir.strip()
    return str(Path.home() / DEFAULT_PROJECT_DATA_DIR)


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    """Path to the test resources directory."""
    current_dir = Path(__file__).parent
    return current_dir / "resources"


class LanguageParamRequest:
    param: Language


def get_repo_path(language: Language) -> Path:
    return Path(__file__).parent / "resources" / "repos" / language / "test_repo"


def _create_ls(
    language: Language,
    repo_path: str | None = None,
    ignored_paths: list[str] | None = None,
    trace_lsp_communication: bool = False,
) -> SolidLanguageServer:
    ignored_paths = ignored_paths or []
    if repo_path is None:
        repo_path = str(get_repo_path(language))

    config = LanguageServerConfig(
        code_language=language,
        ignored_paths=ignored_paths,
        trace_lsp_communication=trace_lsp_communication,
    )
    return SolidLanguageServer.create(
        config,
        repo_path,
        solidlsp_settings=SolidLSPSettings(
            solidlsp_dir=_get_user_data_dir(),
            project_data_relative_path=DEFAULT_PROJECT_DATA_DIR,
        ),
    )


@contextmanager
def start_ls_context(
    language: Language,
    repo_path: str | None = None,
    ignored_paths: list[str] | None = None,
    trace_lsp_communication: bool = False,
) -> Iterator[SolidLanguageServer]:
    ls = _create_ls(language, repo_path, ignored_paths, trace_lsp_communication)
    log.info(f"Starting language server for {language} {repo_path}")
    ls.start()
    try:
        log.info(f"Language server started for {language} {repo_path}")
        yield ls
    finally:
        log.info(f"Stopping language server for {language} {repo_path}")
        try:
            ls.stop(shutdown_timeout=5)
        except Exception as e:
            log.warning(f"Warning: Error stopping language server: {e}")
            if hasattr(ls, "server") and hasattr(ls.server, "process"):
                try:
                    ls.server.process.terminate()
                except Exception:
                    pass


@contextmanager
def start_default_ls_context(language: Language) -> Iterator[SolidLanguageServer]:
    with start_ls_context(language) as ls:
        yield ls


@pytest.fixture(scope="session")
def repo_path(request: LanguageParamRequest) -> Path:
    """Get the repository path for a specific language.

    This fixture requires a language parameter via pytest.mark.parametrize.
    """
    if not hasattr(request, "param"):
        raise ValueError("Language parameter must be provided via pytest.mark.parametrize")
    language = request.param
    return get_repo_path(language)


@pytest.fixture(scope="module")
def language_server(request: LanguageParamRequest) -> Iterator[SolidLanguageServer]:
    """Create a language server instance configured for the specified language.

    This fixture requires a language parameter via pytest.mark.parametrize.
    """
    if not hasattr(request, "param"):
        raise ValueError("Language parameter must be provided via pytest.mark.parametrize")
    language = request.param
    with start_default_ls_context(language) as ls:
        yield ls


is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def _determine_disabled_languages() -> list[Language]:
    """Determine which language tests should be disabled."""
    result: list[Language] = []

    if not is_clojure_cli_available():
        result.append(Language.CLOJURE)

    return result


_disabled_languages = _determine_disabled_languages()


def language_tests_enabled(language: Language) -> bool:
    """Check if tests for the given language are enabled."""
    return language not in _disabled_languages

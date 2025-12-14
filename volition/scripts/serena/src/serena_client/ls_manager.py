"""Language server factory and manager for serena_client.

This module provides a self-contained language server management layer
that doesn't depend on the full serena package.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)

# Default directory name for project-local data
DEFAULT_PROJECT_DATA_DIR = ".serena"


def _get_user_data_dir() -> str:
    """Get the user data directory for language server caches etc.

    Uses SERENA_HOME env var if set, otherwise ~/.serena
    """
    home_dir = os.getenv("SERENA_HOME")
    if home_dir and home_dir.strip():
        return home_dir.strip()
    return str(Path.home() / DEFAULT_PROJECT_DATA_DIR)


class LanguageServerFactory:
    """Factory for creating language server instances."""

    def __init__(
        self,
        project_root: str,
        encoding: str = "utf-8",
        ignored_patterns: list[str] | None = None,
        ls_timeout: float | None = None,
        ls_specific_settings: dict | None = None,
        trace_lsp_communication: bool = False,
        user_data_dir: str | None = None,
        project_data_dir: str = DEFAULT_PROJECT_DATA_DIR,
    ):
        """Create a language server factory.

        Args:
            project_root: Root path of the project
            encoding: File encoding (default: utf-8)
            ignored_patterns: Patterns for files to ignore
            ls_timeout: Timeout for language server operations
            ls_specific_settings: Language-specific settings
            trace_lsp_communication: Whether to trace LSP messages
            user_data_dir: User data directory (default: ~/.serena)
            project_data_dir: Project-local data dir name (default: .serena)

        """
        self.project_root = project_root
        self.encoding = encoding
        self.ignored_patterns = ignored_patterns or []
        self.ls_timeout = ls_timeout
        self.ls_specific_settings = ls_specific_settings
        self.trace_lsp_communication = trace_lsp_communication
        self.user_data_dir = user_data_dir or _get_user_data_dir()
        self.project_data_dir = project_data_dir

    def create_language_server(self, language: Language) -> SolidLanguageServer:
        """Create and return a language server for the given language."""
        ls_config = LanguageServerConfig(
            code_language=language,
            ignored_paths=self.ignored_patterns,
            trace_lsp_communication=self.trace_lsp_communication,
            encoding=self.encoding,
        )

        log.info(f"Creating language server for {self.project_root}, language={language}")
        return SolidLanguageServer.create(
            ls_config,
            self.project_root,
            timeout=self.ls_timeout,
            solidlsp_settings=SolidLSPSettings(
                solidlsp_dir=self.user_data_dir,
                project_data_relative_path=self.project_data_dir,
                ls_specific_settings=self.ls_specific_settings or {},
            ),
        )


class LanguageServerManager:
    """Manages one or more language servers for a project."""

    def __init__(
        self,
        language_servers: dict[Language, SolidLanguageServer],
        language_server_factory: LanguageServerFactory | None = None,
    ) -> None:
        """Create a language server manager.

        Args:
            language_servers: Mapping from language to started language server
            language_server_factory: Optional factory for dynamic server creation

        """
        self._language_servers = language_servers
        self._language_server_factory = language_server_factory
        self._default_language_server = next(iter(language_servers.values()))
        self._root_path = self._default_language_server.repository_root_path

    @staticmethod
    def from_languages(
        languages: list[Language],
        factory: LanguageServerFactory,
    ) -> LanguageServerManager:
        """Create a manager with language servers for the given languages.

        Language servers are started in parallel threads for efficiency.

        Args:
            languages: Languages for which to spawn language servers
            factory: Factory for language server creation

        Returns:
            Configured LanguageServerManager instance

        """
        language_servers: dict[Language, SolidLanguageServer] = {}
        threads: list[threading.Thread] = []
        exceptions: dict[Language, Exception] = {}
        lock = threading.Lock()

        def start_language_server(language: Language) -> None:
            try:
                log.info(f"Starting language server for {language.value}...")
                language_server = factory.create_language_server(language)
                language_server.start()
                if not language_server.is_running():
                    raise RuntimeError(f"Failed to start language server for {language.value}")
                with lock:
                    language_servers[language] = language_server
                log.info(f"Language server for {language.value} started successfully")
            except Exception as e:
                log.error(f"Error starting language server for {language.value}: {e}")
                with lock:
                    exceptions[language] = e

        # Start language servers in parallel
        for language in languages:
            thread = threading.Thread(
                target=start_language_server,
                args=(language,),
                name=f"StartLS:{language.value}",
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # If any server failed, stop all and raise
        if exceptions:
            for ls in language_servers.values():
                ls.stop()
            failure_messages = "\n".join(f"{lang.value}: {e}" for lang, e in exceptions.items())
            raise RuntimeError(f"Failed to start language servers:\n{failure_messages}")

        return LanguageServerManager(language_servers, factory)

    def get_root_path(self) -> str:
        """Get the project root path."""
        return self._root_path

    def get_default_language_server(self) -> SolidLanguageServer:
        """Get the default language server."""
        return self._ensure_functional_ls(self._default_language_server)

    def _ensure_functional_ls(self, ls: SolidLanguageServer) -> SolidLanguageServer:
        """Ensure a language server is running, restarting if needed."""
        if not ls.is_running():
            log.warning(f"Language server for {ls.language} not running; restarting...")
            ls = self.restart_language_server(ls.language)
        return ls

    def get_language_server_for_path(self, relative_path: str) -> SolidLanguageServer | None:
        """Get the appropriate language server for a file path.

        Args:
            relative_path: Path relative to project root

        Returns:
            Language server for the file, or None if no suitable server found

        """
        ls: SolidLanguageServer | None = None
        if len(self._language_servers) > 1:
            for candidate in self._language_servers.values():
                if not candidate.is_ignored_path(relative_path, ignore_unsupported_files=True):
                    ls = candidate
                    break
        if ls is None:
            ls = self._default_language_server
        return self._ensure_functional_ls(ls)

    def _create_and_start_language_server(self, language: Language) -> SolidLanguageServer:
        """Create and start a new language server."""
        if self._language_server_factory is None:
            raise ValueError(f"No factory available to create language server for {language}")
        language_server = self._language_server_factory.create_language_server(language)
        language_server.start()
        self._language_servers[language] = language_server
        return language_server

    def restart_language_server(self, language: Language) -> SolidLanguageServer:
        """Restart the language server for a given language."""
        if language not in self._language_servers:
            raise ValueError(f"No language server for {language.value} present")
        return self._create_and_start_language_server(language)

    def add_language_server(self, language: Language) -> SolidLanguageServer:
        """Add a new language server dynamically."""
        if language in self._language_servers:
            raise ValueError(f"Language server for {language.value} already present")
        return self._create_and_start_language_server(language)

    def remove_language_server(self, language: Language, save_cache: bool = False) -> None:
        """Remove and stop a language server."""
        if language not in self._language_servers:
            raise ValueError(f"No language server for {language.value} present")
        ls = self._language_servers.pop(language)
        self._stop_language_server(ls, save_cache=save_cache)

    @staticmethod
    def _stop_language_server(
        ls: SolidLanguageServer,
        save_cache: bool = False,
        timeout: float = 2.0,
    ) -> None:
        """Stop a language server."""
        if ls.is_running():
            if save_cache:
                ls.save_cache()
            log.info(f"Stopping language server for {ls.language}...")
            ls.stop(shutdown_timeout=timeout)

    def iter_language_servers(self) -> Iterator[SolidLanguageServer]:
        """Iterate over all managed language servers."""
        for ls in self._language_servers.values():
            yield self._ensure_functional_ls(ls)

    def stop_all(self, save_cache: bool = False, timeout: float = 2.0) -> None:
        """Stop all managed language servers."""
        for ls in list(self._language_servers.values()):
            self._stop_language_server(ls, save_cache=save_cache, timeout=timeout)

    def save_all_caches(self) -> None:
        """Save caches for all managed language servers."""
        for ls in self.iter_language_servers():
            if ls.is_running():
                ls.save_cache()

"""Type definitions for gitingest library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepoInfo:
    """Information about a cloned repository."""

    url: str
    local_path: str
    was_cached: bool


@dataclass(frozen=True, slots=True)
class FileContent:
    """Content of a file from the repository."""

    path: str
    content: str
    exists: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DirectoryTree:
    """Directory tree structure."""

    path: str
    tree: str


@dataclass(frozen=True, slots=True)
class GitIngestError:
    """Error from git operations."""

    message: str
    repo_url: str | None = None

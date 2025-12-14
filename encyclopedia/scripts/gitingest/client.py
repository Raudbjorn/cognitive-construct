"""Git repository ingestion library."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .result import Err, Ok, Result
from .types import DirectoryTree, FileContent, GitIngestError, RepoInfo


def _get_repo_cache_path(repo_url: str) -> str:
    """Generate cache path for a repository URL."""
    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"gitingest_{repo_hash}")


def clone_repo(repo_url: str, force_fresh: bool = False) -> Result[RepoInfo, GitIngestError]:
    """Clone a repository and return the local path.

    Uses SHA256 hash-based caching to reuse previously cloned repos.

    Args:
        repo_url: URL of the Git repository to clone
        force_fresh: If True, remove any cached version and clone fresh

    Returns:
        Result containing RepoInfo on success or GitIngestError on failure
    """
    try:
        import git
    except ImportError:
        return Err(GitIngestError("GitPython not installed. Run: pip install gitpython"))

    cache_path = _get_repo_cache_path(repo_url)

    # Force fresh clone if requested
    if force_fresh and os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

    # Check if we have a valid cached clone
    if os.path.exists(cache_path):
        try:
            repo = git.Repo(cache_path)
            if not repo.bare and repo.remote().url == repo_url:
                return Ok(RepoInfo(url=repo_url, local_path=cache_path, was_cached=True))
        except Exception:
            # Cache is invalid, remove it
            shutil.rmtree(cache_path, ignore_errors=True)

    # Clone fresh
    os.makedirs(cache_path, exist_ok=True)
    try:
        git.Repo.clone_from(repo_url, cache_path)
        return Ok(RepoInfo(url=repo_url, local_path=cache_path, was_cached=False))
    except Exception as e:
        shutil.rmtree(cache_path, ignore_errors=True)
        return Err(GitIngestError(f"Failed to clone repository: {e}", repo_url))


def get_directory_tree(
    path: str,
    prefix: str = "",
    exclude_patterns: list[str] | None = None,
) -> Result[DirectoryTree, GitIngestError]:
    """Generate a tree-like directory structure string.

    Args:
        path: Root directory path
        prefix: Prefix for formatting (used internally for recursion)
        exclude_patterns: Patterns to exclude (defaults to [".git"])

    Returns:
        Result containing DirectoryTree on success or GitIngestError on failure
    """
    if not os.path.isdir(path):
        return Err(GitIngestError(f"Path is not a directory: {path}"))

    exclude = exclude_patterns or [".git"]

    def build_tree(dir_path: str, prefix: str) -> str:
        output = ""
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return output

        # Filter excluded patterns
        entries = [e for e in entries if not any(e.startswith(p) for p in exclude)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            current_prefix = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            next_prefix = "    " if is_last else "\u2502   "

            entry_path = os.path.join(dir_path, entry)
            output += prefix + current_prefix + entry + "\n"

            if os.path.isdir(entry_path):
                output += build_tree(entry_path, prefix + next_prefix)

        return output

    tree = build_tree(path, prefix)
    return Ok(DirectoryTree(path=path, tree=tree))


def read_file(repo_path: str, file_path: str) -> Result[FileContent, GitIngestError]:
    """Read a single file from a repository.

    Args:
        repo_path: Path to the cloned repository
        file_path: Path to the file relative to repo root

    Returns:
        Result containing FileContent on success or GitIngestError on failure
    """
    full_path = os.path.join(repo_path, file_path)

    if not os.path.isfile(full_path):
        return Ok(FileContent(path=file_path, content="", exists=False, error="File not found"))

    try:
        with open(full_path, encoding="utf-8") as f:
            content = f.read()
        return Ok(FileContent(path=file_path, content=content))
    except UnicodeDecodeError:
        return Ok(
            FileContent(path=file_path, content="", exists=True, error="Binary or non-UTF-8 file")
        )
    except Exception as e:
        return Err(GitIngestError(f"Error reading file {file_path}: {e}"))


def read_files(
    repo_path: str,
    file_paths: list[str],
) -> Result[list[FileContent], GitIngestError]:
    """Read multiple files from a repository.

    Args:
        repo_path: Path to the cloned repository
        file_paths: List of file paths relative to repo root

    Returns:
        Result containing list of FileContent on success or GitIngestError on failure
    """
    results = []
    for file_path in file_paths:
        result = read_file(repo_path, file_path)
        if result.is_err():
            return result
        results.append(result.value)
    return Ok(results)


@dataclass
class GitIngestClient:
    """Client for ingesting Git repositories.

    Usage:
        client = GitIngestClient()

        # Clone and get tree
        result = client.ingest("https://github.com/user/repo")
        if result.is_ok():
            repo = result.value
            tree = client.get_tree(repo.local_path)
            files = client.read_files(repo.local_path, ["README.md", "src/main.py"])
    """

    cache_dir: str | None = None

    def clone(self, repo_url: str, force_fresh: bool = False) -> Result[RepoInfo, GitIngestError]:
        """Clone a repository."""
        return clone_repo(repo_url, force_fresh)

    def get_tree(
        self,
        path: str,
        exclude_patterns: list[str] | None = None,
    ) -> Result[DirectoryTree, GitIngestError]:
        """Get directory tree for a path."""
        return get_directory_tree(path, exclude_patterns=exclude_patterns)

    def read_file(self, repo_path: str, file_path: str) -> Result[FileContent, GitIngestError]:
        """Read a single file from a repository."""
        return read_file(repo_path, file_path)

    def read_files(
        self,
        repo_path: str,
        file_paths: list[str],
    ) -> Result[list[FileContent], GitIngestError]:
        """Read multiple files from a repository."""
        return read_files(repo_path, file_paths)

    def ingest(
        self,
        repo_url: str,
        file_paths: list[str] | None = None,
        include_tree: bool = True,
        force_fresh: bool = False,
    ) -> Result[dict, GitIngestError]:
        """Full ingestion: clone repo, get tree, and optionally read files.

        Args:
            repo_url: URL of the Git repository
            file_paths: Optional list of files to read
            include_tree: Whether to include directory tree (default: True)
            force_fresh: Force fresh clone (default: False)

        Returns:
            Result containing dict with repo info, tree, and file contents
        """
        # Clone
        clone_result = self.clone(repo_url, force_fresh)
        if clone_result.is_err():
            return clone_result

        repo = clone_result.value
        result: dict = {"repo": repo}

        # Get tree
        if include_tree:
            tree_result = self.get_tree(repo.local_path)
            if tree_result.is_ok():
                result["tree"] = tree_result.value

        # Read files
        if file_paths:
            files_result = self.read_files(repo.local_path, file_paths)
            if files_result.is_ok():
                result["files"] = files_result.value

        return Ok(result)


# Convenience function
def ingest(
    repo_url: str,
    file_paths: list[str] | None = None,
    include_tree: bool = True,
) -> Result[dict, GitIngestError]:
    """Ingest a Git repository: clone, get tree, and optionally read files.

    Convenience function that creates a client for single use.
    """
    client = GitIngestClient()
    return client.ingest(repo_url, file_paths, include_tree)

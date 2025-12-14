"""
gitingest - Python library for Git repository ingestion and analysis

Usage:
    from gitingest import GitIngestClient, ingest

    # Quick ingestion
    result = ingest("https://github.com/user/repo")
    if result.is_ok():
        print(result.value["tree"].tree)

    # Full client usage
    client = GitIngestClient()

    # Clone
    result = client.clone("https://github.com/user/repo")
    if result.is_ok():
        repo = result.value
        print(f"Cloned to: {repo.local_path}")

        # Get directory tree
        tree = client.get_tree(repo.local_path)
        if tree.is_ok():
            print(tree.value.tree)

        # Read specific files
        files = client.read_files(repo.local_path, ["README.md", "src/main.py"])
        if files.is_ok():
            for f in files.value:
                print(f"{f.path}: {len(f.content)} bytes")
"""

from .client import (
    GitIngestClient,
    clone_repo,
    get_directory_tree,
    ingest,
    read_file,
    read_files,
)
from .types import DirectoryTree, FileContent, GitIngestError, RepoInfo
from .result import Result, Ok, Err

__all__ = [
    "GitIngestClient",
    "clone_repo",
    "get_directory_tree",
    "read_file",
    "read_files",
    "ingest",
    "RepoInfo",
    "DirectoryTree",
    "FileContent",
    "GitIngestError",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

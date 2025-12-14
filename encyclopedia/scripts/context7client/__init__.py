"""
context7client - Python client library for Context7 API

Context7 provides up-to-date documentation and code examples for libraries.

Usage:
    from context7client import Context7Client, search_library, get_docs

    # Client-based usage
    client = Context7Client(api_key="ctx7sk...")  # or set CONTEXT7_API_KEY env var

    # Search for a library
    result = await client.search_library("react")
    if result.is_ok():
        for lib in result.value.results:
            print(f"{lib.title}: {lib.id}")

    # Get documentation (code examples)
    result = await client.get_docs("/facebook/react", topic="hooks")
    if result.is_ok():
        for snippet in result.value.snippets:
            print(f"{snippet.code_title}")
            for ex in snippet.code_list:
                print(f"  {ex.language}: {ex.code[:50]}...")

    # Get documentation (conceptual guides)
    result = await client.get_docs("/facebook/react", topic="hooks", mode="info")

    # Resolve library name to ID
    lib_id = await client.resolve_library_id("svelte")
"""

from .client import Context7Client, get_docs, search_library
from .types import (
    APIResponseMetadata,
    CodeDocsResponse,
    CodeExample,
    CodeSnippet,
    Context7Error,
    DocsFormat,
    DocsMode,
    GetDocsOptions,
    InfoDocsResponse,
    InfoSnippet,
    LibraryState,
    Pagination,
    SearchLibraryResponse,
    SearchResult,
    TextDocsResponse,
)
from .result import Result, Ok, Err

__all__ = [
    "Context7Client",
    "search_library",
    "get_docs",
    "SearchLibraryResponse",
    "SearchResult",
    "APIResponseMetadata",
    "CodeDocsResponse",
    "InfoDocsResponse",
    "TextDocsResponse",
    "CodeSnippet",
    "CodeExample",
    "InfoSnippet",
    "Pagination",
    "GetDocsOptions",
    "DocsMode",
    "DocsFormat",
    "LibraryState",
    "Context7Error",
    "Result",
    "Ok",
    "Err",
]

__version__ = "1.0.0"

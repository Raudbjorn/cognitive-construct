#!/usr/bin/env python3
"""
Encyclopedia Skill - Knowledge retrieval for the Cognitive Construct.

Aggregates multiple knowledge sources (Context7, Exa, Perplexity, mcp-git-ingest)
using native Python client libraries for backend queries.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable

# Import client libraries from sibling packages
sys.path.insert(0, str(Path(__file__).parent))

from context7client import Context7Client
from exaclient import ExaClient, WebSearchOptions, CodeSearchOptions
from perplexity import PerplexityClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).parent
PROJECT_ROOT = SKILL_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env.local"
ENCYCLOPEDIA_DIR = Path.home() / ".encyclopedia"
CACHE_DIR = ENCYCLOPEDIA_DIR / "cache"
SESSIONS_DIR = ENCYCLOPEDIA_DIR / "sessions"
SOURCE_CONFIG_FILE = SKILL_DIR / "resources" / "source_config.json"


def load_source_config() -> dict[str, Any]:
    """Load routing and source priority config."""
    default_config = {
        "sources": {
            "context7": {"priority": 100},
            "exa": {"priority": 80},
            "perplexity": {"priority": 70},
            "mcp_git_ingest": {"priority": 60},
            "kagi": {"priority": 50},
            "searxng": {"priority": 40},
            "codegraph": {"priority": 30},
        },
        "routing": {
            "library_docs": ["context7", "exa"],
            "general_search": ["exa", "perplexity"],
            "code_context": ["mcp_git_ingest", "exa", "codegraph"],
            "repository": ["mcp_git_ingest"],
        },
        "deduplication": {"similarity_threshold": 0.85, "content_preview_length": 200},
        "timeouts": {"query_timeout_seconds": 10.0, "total_timeout_seconds": 30.0},
    }

    if not SOURCE_CONFIG_FILE.exists():
        return default_config

    try:
        with open(SOURCE_CONFIG_FILE) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default_config

    return config


SOURCE_CONFIG = load_source_config()

SOURCE_PRIORITY = {
    name: details.get("priority", 0) for name, details in SOURCE_CONFIG.get("sources", {}).items()
}
KNOWN_SOURCES = set(SOURCE_PRIORITY.keys())

SOURCE_ROUTING = SOURCE_CONFIG.get("routing", {})
SIMILARITY_THRESHOLD = SOURCE_CONFIG.get("deduplication", {}).get("similarity_threshold", 0.85)
QUERY_TIMEOUT = SOURCE_CONFIG.get("timeouts", {}).get("query_timeout_seconds", 10.0)
TOTAL_TIMEOUT = SOURCE_CONFIG.get("timeouts", {}).get("total_timeout_seconds", 30.0)
OPTIONAL_SOURCES = {"kagi", "searxng", "codegraph"}


class ErrorCode(Enum):
    CONFIG_ERROR = 1
    NOT_FOUND = 2
    BACKEND_UNAVAILABLE = 3
    INTERNAL_ERROR = 4


REPO_HINT_PATTERN = re.compile(r'repo:([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)')
TRUTHY = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Ensure config directories exist."""
    ENCYCLOPEDIA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_env_local() -> dict[str, str]:
    """Load environment variables from .env.local file."""
    env_vars: dict[str, str] = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip().strip("\"'")
                    env_vars[key.strip()] = value
                    os.environ[key.strip()] = value
    return env_vars


def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    sanitized = re.sub(r'[<>]', '', text)
    return sanitized[:2000]


def parse_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variables."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def output_json(data: dict[str, Any], exit_code: int = 0) -> None:
    """Output JSON response and exit with code."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(exit_code)


def output_error(code: ErrorCode, message: str) -> None:
    """Output error response and exit."""
    output_json({"status": "error", "code": code.value, "message": message}, code.value)


def get_cache_key(query: str, source: str) -> str:
    """Generate cache key for a query."""
    content = f"{source}:{query}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_cached(query: str, source: str, max_age: timedelta = timedelta(hours=1)) -> dict | None:
    """Get cached response if available and not expired."""
    cache_key = get_cache_key(query, source)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
            if datetime.now() - cached_time < max_age:
                return cached.get("data")
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return None


def set_cache(query: str, source: str, data: dict) -> None:
    """Cache a response."""
    ensure_dirs()
    cache_key = get_cache_key(query, source)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    with open(cache_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

class FeatureFlags:
    """Feature toggles for optional sources."""

    def __init__(self) -> None:
        self._flags = {
            "context7": parse_bool_env("ENCYCLOPEDIA_ENABLE_CONTEXT7", True),
            "kagi": parse_bool_env("ENCYCLOPEDIA_ENABLE_KAGI", False),
            "searxng": parse_bool_env("ENCYCLOPEDIA_ENABLE_SEARXNG", False),
            "codegraph": parse_bool_env("ENCYCLOPEDIA_ENABLE_CODEGRAPH", False),
        }

    def enabled(self, source: str) -> bool:
        return self._flags.get(source, True)


@dataclass
class Credentials:
    """Manages API credentials with validation."""
    exa_key: str | None = None
    perplexity_key: str | None = None
    context7_key: str | None = None
    kagi_key: str | None = None
    searxng_url: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    @classmethod
    def load(cls) -> "Credentials":
        """Load credentials from environment."""
        load_env_local()
        return cls(
            exa_key=os.environ.get("EXA_API_KEY"),
            perplexity_key=os.environ.get("PERPLEXITY_API_KEY"),
            context7_key=os.environ.get("CONTEXT7_API_KEY"),
            kagi_key=os.environ.get("KAGI_API_KEY"),
            searxng_url=os.environ.get("SEARXNG_URL"),
            neo4j_uri=os.environ.get("NEO4J_URI"),
            neo4j_user=os.environ.get("NEO4J_USERNAME"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD"),
        )

    def has_any_search(self) -> bool:
        """Check if any search provider is configured."""
        return bool(self.exa_key or self.perplexity_key or self.kagi_key or self.searxng_url)

    def is_source_available(
        self,
        source: str,
        feature_flags: FeatureFlags | None = None,
    ) -> tuple[bool, str | None]:
        """Check if a specific source can be used."""
        if feature_flags and not feature_flags.enabled(source):
            return False, "feature_disabled"

        if source == "context7":
            # Context7 works without API key (rate limited)
            return True, None
        if source == "exa":
            return bool(self.exa_key), "missing_credentials"
        if source == "perplexity":
            return bool(self.perplexity_key), "missing_credentials"
        if source == "kagi":
            return bool(self.kagi_key), "missing_credentials"
        if source == "searxng":
            return bool(self.searxng_url), "missing_credentials"
        if source == "codegraph":
            ready = bool(self.neo4j_uri and self.neo4j_user and self.neo4j_password)
            return ready, "missing_credentials"
        if source == "mcp_git_ingest":
            # Always available (uses HTTP)
            return True, None
        return False, "unknown_source"

    def get_available_sources(self, feature_flags: FeatureFlags | None = None) -> list[str]:
        """Get list of sources with valid credentials and enabled flags."""
        available = []
        for source in KNOWN_SOURCES | {"mcp_git_ingest"}:
            ok, _ = self.is_source_available(source, feature_flags)
            if ok:
                available.append(source)
        return available

    def validate(self) -> tuple[bool, str]:
        """Validate that at least one search provider is available."""
        if not self.has_any_search():
            missing = []
            if not self.exa_key:
                missing.append("EXA_API_KEY")
            if not self.perplexity_key:
                missing.append("PERPLEXITY_API_KEY")
            return False, f"No search providers configured. Set at least one of: {', '.join(sorted(set(missing)))}"
        return True, ""


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

def classify_query(query: str) -> str:
    """Classify query into a type for routing."""
    query_lower = query.lower()

    # Explicit type hints
    if query_lower.startswith("doc:") or query_lower.startswith("docs:"):
        return "library_docs"
    if query_lower.startswith("code:"):
        return "code_context"
    if query_lower.startswith("web:"):
        return "general_search"

    # URL patterns -> general search
    if re.search(r'https?://', query):
        return "general_search"

    # Code patterns -> code context
    code_patterns = [r'\bdef\s+\w+', r'\bclass\s+\w+', r'\bfunction\s+\w+',
                     r'\bimport\s+', r'\bfrom\s+\w+\s+import']
    if any(re.search(p, query) for p in code_patterns):
        return "code_context"

    # Repository hint -> code context
    if REPO_HINT_PATTERN.search(query_lower):
        return "code_context"

    # Time-sensitive keywords -> general search
    time_keywords = ["latest", "current", "2024", "2025", "recent", "news", "update"]
    if any(kw in query_lower for kw in time_keywords):
        return "general_search"

    # Library/framework keywords -> library docs
    library_keywords = ["how to use", "api", "documentation", "example", "tutorial",
                        "react", "vue", "angular", "django", "flask", "fastapi",
                        "express", "nextjs", "typescript", "python", "rust", "go"]
    if any(kw in query_lower for kw in library_keywords):
        return "library_docs"

    # Default to library docs
    return "library_docs"


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    content: str
    url: str | None = None
    source: str = ""
    relevance: float = 0.0
    timestamp: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content[:500] if len(self.content) > 500 else self.content,
            "url": self.url,
            "source": self.source,
            "relevance": self.relevance,
        }


def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using sequence matching."""
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    return SequenceMatcher(None, t1, t2).ratio()


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate results based on semantic similarity."""
    if not results:
        return []

    # Sort by source priority (higher priority first)
    sorted_results = sorted(
        results,
        key=lambda r: SOURCE_PRIORITY.get(r.source, 0),
        reverse=True
    )

    deduplicated: list[SearchResult] = []
    for result in sorted_results:
        is_duplicate = False
        for existing in deduplicated:
            title_sim = semantic_similarity(result.title, existing.title)
            content_sim = semantic_similarity(result.content[:200], existing.content[:200])

            if title_sim > SIMILARITY_THRESHOLD or content_sim > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(result)

    return deduplicated


def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    """Rank results by relevance and recency."""
    def score(r: SearchResult) -> tuple[float, float]:
        recency = 0.0
        if r.timestamp:
            days_old = (datetime.now() - r.timestamp).days
            recency = max(0, 1 - days_old / 365)
        return (r.relevance, recency)

    return sorted(results, key=score, reverse=True)


def extract_repo_hint(query: str) -> tuple[str | None, str]:
    """Extract repo hint (repo:owner/name) and return cleaned query."""
    match = REPO_HINT_PATTERN.search(query)
    if not match:
        return None, query

    repo = match.group(1)
    cleaned = (query[:match.start()] + query[match.end():]).strip()
    return repo, cleaned or query


def is_optional_source(source: str) -> bool:
    return source in OPTIONAL_SOURCES


class DegradationTracker:
    """Collect degraded source metadata."""

    def __init__(self) -> None:
        self.missing: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

    def add_missing(self, source: str, reason: str, optional: bool) -> None:
        self.missing.append(
            {"source": source, "reason": reason, "optional": optional}
        )

    def add_error(self, source: str, message: str, optional: bool) -> None:
        self.errors.append(
            {"source": source, "message": message, "optional": optional}
        )

    def summary(self) -> dict[str, Any]:
        return {"missing": self.missing, "errors": self.errors}

    @property
    def is_degraded(self) -> bool:
        return bool(self.missing or self.errors)


async def run_source_task(source: str, coro: Awaitable[list[SearchResult]]) -> tuple[str, list[SearchResult] | None, Exception | None]:
    """Wrap a source coroutine to capture exceptions."""
    try:
        data = await coro
        return source, data, None
    except Exception as exc:
        return source, None, exc


# ---------------------------------------------------------------------------
# Source Integrations (Native Python Clients)
# ---------------------------------------------------------------------------

async def query_context7(query: str, creds: Credentials) -> list[SearchResult]:
    """Query Context7 for library documentation using native client."""
    results: list[SearchResult] = []

    # Check cache
    cached = get_cached(query, "context7")
    if cached:
        return [SearchResult(**r) for r in cached]  # source already in cached data

    client = Context7Client(api_key=creds.context7_key)
    search_result = await client.search_library(query)

    if search_result.is_err():
        return results

    for match in search_result.value.results[:3]:
        results.append(SearchResult(
            title=match.title or query,
            content=match.description or "",
            url=f"https://context7.com{match.id}",
            source="context7",
            relevance=0.9 if match.benchmark_score and match.benchmark_score > 70 else 0.7,
            metadata={
                "library_id": match.id,
                "snippets": match.total_snippets,
                "trust_score": match.trust_score,
            }
        ))

    if results:
        set_cache(query, "context7", [r.to_dict() for r in results])
    return results


async def query_exa(query: str, creds: Credentials) -> list[SearchResult]:
    """Query Exa for web and code search using native client."""
    results: list[SearchResult] = []

    if not creds.exa_key:
        return results

    cached = get_cached(query, "exa")
    if cached:
        return [SearchResult(**r) for r in cached]  # source already in cached data

    client = ExaClient(api_key=creds.exa_key)
    search_result = client.web_search(query, WebSearchOptions(num_results=5))

    if search_result.is_ok():
        context = search_result.value.context
        if context:
            results.append(SearchResult(
                title=f"Exa: {query[:50]}",
                content=context,
                source="exa",
                relevance=0.8,
            ))
            # Also include individual results if available
            for item in search_result.value.response.results[:3]:
                if item.text:
                    results.append(SearchResult(
                        title=item.title or "Exa Result",
                        content=item.text[:500],
                        url=item.url,
                        source="exa",
                        relevance=item.score or 0.7,
                    ))

    if results:
        set_cache(query, "exa", [r.to_dict() for r in results])
    return results


async def query_exa_code(query: str, creds: Credentials) -> list[SearchResult]:
    """Query Exa for code-specific search."""
    results: list[SearchResult] = []

    if not creds.exa_key:
        return results

    cached = get_cached(query, "exa_code")
    if cached:
        return [SearchResult(**r) for r in cached]  # source already in cached data

    client = ExaClient(api_key=creds.exa_key)
    search_result = client.code_search(query, CodeSearchOptions(tokens=5000))

    if search_result.is_ok():
        content = search_result.value.content
        if content:
            results.append(SearchResult(
                title=f"Exa Code: {query[:50]}",
                content=content[:2000],
                source="exa",
                relevance=0.85,
                metadata={"type": "code"}
            ))

    if results:
        set_cache(query, "exa_code", [r.to_dict() for r in results])
    return results


async def query_perplexity(query: str, creds: Credentials) -> list[SearchResult]:
    """Query Perplexity for AI-powered search using native client."""
    results: list[SearchResult] = []

    if not creds.perplexity_key:
        return results

    cached = get_cached(query, "perplexity")
    if cached:
        return [SearchResult(**r) for r in cached]  # source already in cached data

    client = PerplexityClient(api_key=creds.perplexity_key)
    ask_result = await client.ask(query)

    if ask_result.is_ok():
        content = ask_result.value
        if content:
            results.append(SearchResult(
                title=f"Perplexity: {query[:50]}",
                content=content,
                source="perplexity",
                relevance=0.85,
            ))

    if results:
        set_cache(query, "perplexity", [r.to_dict() for r in results])
    return results


async def query_git_ingest(repo_path: str, query: str) -> list[SearchResult]:
    """Query mcp-git-ingest for repository analysis.

    Uses HTTP API to fetch repository structure.
    """
    import httpx

    results: list[SearchResult] = []

    # Parse repo path
    repo_match = re.match(r'(?:github\.com/)?([^/]+/[^/]+)', repo_path)
    if not repo_match:
        return results

    repo = repo_match.group(1)
    repo_url = f"https://github.com/{repo}"

    # Use gitingest API for repo summary
    try:
        async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
            # Try gitingest.com API
            response = await client.get(
                f"https://gitingest.com/{repo}",
                headers={"Accept": "text/plain"},
                follow_redirects=True,
            )
            if response.status_code == 200:
                content = response.text[:3000]
                results.append(SearchResult(
                    title=f"{repo} - Repository Analysis",
                    content=content,
                    url=repo_url,
                    source="mcp_git_ingest",
                    relevance=0.8,
                    metadata={"repo": repo}
                ))
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Main Query Orchestration
# ---------------------------------------------------------------------------

async def execute_search(query: str, sources: list[str] | None = None, limit: int = 5) -> dict:
    """Execute search across multiple sources with parallel queries."""
    creds = Credentials.load()
    feature_flags = FeatureFlags()
    degradation = DegradationTracker()

    # Validate credentials
    valid, msg = creds.validate()
    if not valid:
        return {
            "status": "error",
            "code": ErrorCode.CONFIG_ERROR.value,
            "message": msg,
            "degraded": True,
            "degradation": degradation.summary(),
        }

    # Determine sources to query
    query_type = classify_query(query)
    repo_hint, cleaned_query = extract_repo_hint(query)
    effective_query = cleaned_query or query
    requested_sources: list[str] | None = None

    if sources:
        normalized = [s.strip().lower() for s in sources if s.strip()]
        unknown = [s for s in normalized if s not in KNOWN_SOURCES]
        if unknown:
            return {
                "status": "error",
                "code": ErrorCode.CONFIG_ERROR.value,
                "message": f"Unknown sources: {', '.join(sorted(set(unknown)))}",
                "degraded": False,
                "degradation": degradation.summary(),
            }
        requested_sources = normalized

    # Use routing table
    routed_sources = requested_sources or SOURCE_ROUTING.get(query_type, ["exa", "perplexity"])
    target_sources: list[str] = []

    for source in routed_sources:
        is_available, reason = creds.is_source_available(source, feature_flags)
        if is_available:
            target_sources.append(source)
        else:
            degradation.add_missing(source, reason or "unavailable", is_optional_source(source))

    # Special handling for repo-dependent sources
    if repo_hint is None:
        target_sources = [
            s for s in target_sources if s not in {"mcp_git_ingest", "codegraph"}
        ]
        for src in ("mcp_git_ingest", "codegraph"):
            if src in routed_sources:
                degradation.add_missing(
                    src,
                    "repository hint missing (use repo:owner/name)",
                    is_optional_source(src),
                )
    else:
        if "mcp_git_ingest" not in target_sources and "mcp_git_ingest" in routed_sources:
            target_sources.append("mcp_git_ingest")

    if not target_sources:
        return {
            "status": "error",
            "code": ErrorCode.BACKEND_UNAVAILABLE.value,
            "message": "No search sources available",
            "degraded": degradation.is_degraded,
            "degradation": degradation.summary(),
        }

    # Execute queries in parallel
    tasks: list[asyncio.Task] = []
    for source in target_sources:
        if source == "context7":
            coro = query_context7(effective_query, creds)
        elif source == "exa":
            if query_type == "code_context":
                coro = query_exa_code(effective_query, creds)
            else:
                coro = query_exa(effective_query, creds)
        elif source == "perplexity":
            coro = query_perplexity(effective_query, creds)
        elif source == "mcp_git_ingest" and repo_hint:
            coro = query_git_ingest(repo_hint, effective_query)
        else:
            continue
        task = asyncio.create_task(run_source_task(source, coro))
        setattr(task, "source_name", source)
        tasks.append(task)

    done_results: list[tuple[str, list[SearchResult] | None, Exception | None]] = []
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=TOTAL_TIMEOUT)
        for task in done:
            done_results.append(task.result())
        for task in pending:
            source_name = getattr(task, "source_name", None)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if source_name:
                degradation.add_error(
                    source_name,
                    f"Timed out after {TOTAL_TIMEOUT}s",
                    is_optional_source(source_name),
                )

    # Flatten and filter results
    results: list[SearchResult] = []
    sources_used: list[str] = []

    for source_name, data, error in done_results:
        if error:
            degradation.add_error(
                source_name,
                str(error),
                is_optional_source(source_name),
            )
            continue
        if data:
            results.extend(data)
            sources_used.append(source_name)

    if not results:
        return {
            "status": "error",
            "code": ErrorCode.NOT_FOUND.value,
            "message": "No results found",
            "degraded": degradation.is_degraded,
            "degradation": degradation.summary(),
        }

    # Deduplicate and rank
    results = deduplicate_results(results)
    results = rank_results(results)

    # Limit results
    results = results[:limit]

    return {
        "status": "success",
        "query_type": query_type,
        "results": [r.to_dict() for r in results],
        "sources_used": list(dict.fromkeys(sources_used)),
        "degraded": degradation.is_degraded,
        "degradation": degradation.summary(),
    }


async def execute_lookup(topic: str, version: str | None = None) -> dict:
    """Look up documentation for a specific topic."""
    creds = Credentials.load()

    query = f"doc: {topic}"
    if version:
        query += f" version {version}"

    # Query Context7 first (best for library docs)
    results = await query_context7(topic, creds)

    if not results:
        # Fallback to Exa
        results = await query_exa(query, creds)

    if not results:
        return {
            "status": "error",
            "code": ErrorCode.NOT_FOUND.value,
            "message": f"No documentation found for: {topic}"
        }

    result = results[0]
    return {
        "status": "success",
        "topic": topic,
        "version": result.metadata.get("version") or version or "latest",
        "content": result.content,
        "url": result.url,
        "source": result.source
    }


async def execute_code(repo_path: str, query: str, depth: str = "shallow") -> dict:
    """Analyze a code repository."""
    # Parse and validate repo path
    repo_match = re.match(r'(?:github\.com/)?([^/]+/[^/]+)', repo_path)
    if not repo_match:
        return {
            "status": "error",
            "code": ErrorCode.CONFIG_ERROR.value,
            "message": f"Invalid repository path: {repo_path}"
        }

    repo = repo_match.group(1)
    results = await query_git_ingest(repo_path, query)

    if not results:
        return {
            "status": "error",
            "code": ErrorCode.NOT_FOUND.value,
            "message": f"Could not analyze repository: {repo}"
        }

    # Combine results
    analysis_parts = []
    for r in results:
        analysis_parts.append(f"## {r.title}\n\n{r.content}")

    return {
        "status": "success",
        "repository": repo,
        "depth": depth,
        "analysis": "\n\n---\n\n".join(analysis_parts),
        "metadata": results[0].metadata if results else {}
    }


# ---------------------------------------------------------------------------
# CLI Command Handlers
# ---------------------------------------------------------------------------

def handle_search(query: str, sources: str | None, limit: int) -> None:
    """Handle the search command."""
    sanitized = sanitize_input(query)
    if not sanitized:
        output_error(ErrorCode.CONFIG_ERROR, "Query cannot be empty")

    source_list = sources.split(",") if sources else None
    source_list = [s.strip().lower() for s in source_list] if source_list else None
    result = asyncio.run(execute_search(sanitized, source_list, limit))
    output_json(result, result.get("code", 0))


def handle_lookup(topic: str, version: str | None) -> None:
    """Handle the lookup command."""
    sanitized = sanitize_input(topic)
    if not sanitized:
        output_error(ErrorCode.CONFIG_ERROR, "Topic cannot be empty")

    result = asyncio.run(execute_lookup(sanitized, version))
    output_json(result, result.get("code", 0))


def handle_code(repo_path: str, query: str, depth: str) -> None:
    """Handle the code command."""
    sanitized_repo = sanitize_input(repo_path)
    sanitized_query = sanitize_input(query)

    if not sanitized_repo:
        output_error(ErrorCode.CONFIG_ERROR, "Repository path cannot be empty")

    result = asyncio.run(execute_code(sanitized_repo, sanitized_query, depth))
    output_json(result, result.get("code", 0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Encyclopedia Skill - Knowledge retrieval for the Cognitive Construct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  encyclopedia.py search "React useState best practices"
  encyclopedia.py lookup "fastapi" --version latest
  encyclopedia.py code "github.com/owner/repo" "how does auth work"
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SEARCH command
    search_parser = subparsers.add_parser("search", help="Search across knowledge sources")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--sources",
        help="Comma-separated list of sources (context7,exa,perplexity,kagi,searxng,codegraph,mcp_git_ingest)"
    )
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum results (default: 5)")

    # LOOKUP command
    lookup_parser = subparsers.add_parser("lookup", help="Look up documentation for a topic")
    lookup_parser.add_argument("topic", help="Topic to look up (library, API, framework)")
    lookup_parser.add_argument("--version", help="Specific version (default: latest)")

    # CODE command
    code_parser = subparsers.add_parser("code", help="Analyze a code repository")
    code_parser.add_argument("repo_path", help="Repository path (github.com/owner/repo or owner/repo)")
    code_parser.add_argument("query", help="Question about the repository")
    code_parser.add_argument("--depth", choices=["shallow", "deep"], default="shallow",
                            help="Analysis depth (default: shallow)")

    args = parser.parse_args()

    # Ensure directories exist
    ensure_dirs()

    # Route to command handler
    if args.command == "search":
        handle_search(args.query, args.sources, args.limit)
    elif args.command == "lookup":
        handle_lookup(args.topic, args.version)
    elif args.command == "code":
        handle_code(args.repo_path, args.query, args.depth)


if __name__ == "__main__":
    main()

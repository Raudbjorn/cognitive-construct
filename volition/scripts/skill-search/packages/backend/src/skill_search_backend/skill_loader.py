"""Skill loading and parsing functionality."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .config import Config, config_to_dict
from .result import Err, Ok, Result
from .types import LoadError, Skill

logger = logging.getLogger(__name__)


# === Helper Functions ===


def _is_text_file(file_path: Path, text_extensions: tuple[str, ...] | list[str]) -> bool:
    """Check if a file is a text file based on extension."""
    return file_path.suffix.lower() in text_extensions


def _is_image_file(file_path: Path, image_extensions: tuple[str, ...] | list[str]) -> bool:
    """Check if a file is an image based on extension."""
    return file_path.suffix.lower() in image_extensions


def _load_text_file(file_path: Path) -> dict[str, Any] | None:
    """Load a text file and return its metadata."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "type": "text",
            "content": content,
            "size": len(content),
        }
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return None


def _load_image_file(
    file_path: Path, max_size: int, url: str | None = None
) -> dict[str, Any] | None:
    """Load an image file and return its metadata with base64 encoding."""
    try:
        file_size = file_path.stat().st_size

        if file_size > max_size:
            logger.warning(
                f"Image {file_path} exceeds size limit ({file_size} > {max_size}), "
                "storing metadata only"
            )
            result: dict[str, Any] = {
                "type": "image",
                "size": file_size,
                "size_exceeded": True,
            }
            if url:
                result["url"] = url
            return result

        image_data = file_path.read_bytes()
        base64_content = base64.b64encode(image_data).decode("utf-8")

        result = {
            "type": "image",
            "content": base64_content,
            "size": file_size,
        }
        if url:
            result["url"] = url

        return result

    except Exception as e:
        logger.error(f"Error reading image file {file_path}: {e}")
        return None


def _load_documents_from_directory(
    skill_dir: Path,
    text_extensions: tuple[str, ...] | list[str],
    image_extensions: tuple[str, ...] | list[str],
    max_image_size: int,
) -> dict[str, dict[str, Any]]:
    """Load all documents from a skill directory."""
    documents: dict[str, dict[str, Any]] = {}

    for file_path in skill_dir.rglob("*"):
        if file_path.name == "SKILL.md" or file_path.is_dir():
            continue

        try:
            rel_path = str(file_path.relative_to(skill_dir))
        except ValueError:
            continue

        if _is_text_file(file_path, text_extensions):
            doc_data = _load_text_file(file_path)
            if doc_data:
                documents[rel_path] = doc_data

        elif _is_image_file(file_path, image_extensions):
            doc_data = _load_image_file(file_path, max_image_size)
            if doc_data:
                documents[rel_path] = doc_data

    return documents


# === Skill Parsing ===


def parse_skill_md(content: str, source: str) -> Result[Skill, LoadError]:
    """Parse a SKILL.md file and extract skill information.

    Parameters
    ----------
    content : str
        Content of the SKILL.md file.
    source : str
        Origin of the skill (for tracking).

    Returns
    -------
    Result[Skill, LoadError]
        Ok with parsed Skill or Err with LoadError if parsing failed.
    """
    try:
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
        )

        if not frontmatter_match:
            return Err(
                LoadError(
                    message="No YAML frontmatter found",
                    source=source,
                )
            )

        frontmatter_text = frontmatter_match.group(1)
        markdown_body = frontmatter_match.group(2)

        name_match = re.search(r"^name:\s*(.+)$", frontmatter_text, re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", frontmatter_text, re.MULTILINE)

        if not name_match or not desc_match:
            return Err(
                LoadError(
                    message="Missing name or description in frontmatter",
                    source=source,
                )
            )

        name = name_match.group(1).strip().strip("\"'")
        description = desc_match.group(1).strip().strip("\"'")

        return Ok(
            Skill(
                name=name,
                description=description,
                content=markdown_body.strip(),
                source=source,
            )
        )

    except Exception as e:
        return Err(
            LoadError(
                message=f"Error parsing SKILL.md: {e}",
                source=source,
                cause=str(e),
            )
        )


# === Cache Management ===


def _get_document_cache_dir() -> Path:
    """Get document cache directory."""
    cache_dir = Path(tempfile.gettempdir()) / "skill_search_cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path(url: str, branch: str) -> Path:
    """Get cache file path for a GitHub repository."""
    cache_dir = Path(tempfile.gettempdir()) / "skill_search_cache"
    cache_dir.mkdir(exist_ok=True)

    cache_key = f"{url}_{branch}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()

    return cache_dir / f"{hash_key}.json"


def _load_from_cache(
    cache_path: Path, max_age_hours: int = 24
) -> dict[str, Any] | None:
    """Load cached GitHub API response if available and not expired."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        cached_time = datetime.fromisoformat(cache_data["timestamp"])
        if datetime.now() - cached_time > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {cache_path}")
            return None

        logger.info(f"Using cached GitHub API response from {cache_path}")
        return cache_data["tree_data"]

    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def _save_to_cache(cache_path: Path, tree_data: dict[str, Any]) -> None:
    """Save GitHub API response to cache."""
    try:
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "tree_data": tree_data,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        logger.info(f"Saved GitHub API response to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


# === GitHub Document Handling ===


def _get_document_metadata_from_github(
    owner: str,
    repo: str,
    branch: str,
    skill_dir_path: str,
    tree_data: dict[str, Any],
    text_extensions: tuple[str, ...] | list[str],
    image_extensions: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    """Get document metadata from GitHub without fetching content."""
    documents: dict[str, dict[str, Any]] = {}

    for item in tree_data.get("tree", []):
        if item["type"] != "blob":
            continue

        item_path = item["path"]

        if not item_path.startswith(skill_dir_path):
            continue

        if item_path.endswith("/SKILL.md") or item_path == f"{skill_dir_path}/SKILL.md":
            continue

        if skill_dir_path:
            rel_path = item_path[len(skill_dir_path) :].lstrip("/")
        else:
            rel_path = item_path

        if not rel_path:
            continue

        file_ext = Path(item_path).suffix.lower()

        if file_ext in text_extensions:
            documents[rel_path] = {
                "type": "text",
                "size": item.get("size", 0),
                "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item_path}",
                "fetched": False,
            }
        elif file_ext in image_extensions:
            documents[rel_path] = {
                "type": "image",
                "size": item.get("size", 0),
                "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item_path}",
                "fetched": False,
            }

    return documents


def _create_document_fetcher(
    owner: str,
    repo: str,
    branch: str,
    skill_dir_path: str,
    text_extensions: tuple[str, ...] | list[str],
    image_extensions: tuple[str, ...] | list[str],
    max_image_size: int,
) -> Callable[[str], dict[str, Any] | None]:
    """Create a closure that fetches documents on-demand with disk caching."""
    cache_dir = _get_document_cache_dir()

    def fetch_document(doc_path: str) -> dict[str, Any] | None:
        """Fetch a single document with local caching."""
        if skill_dir_path:
            full_path = f"{skill_dir_path}/{doc_path}"
        else:
            full_path = doc_path

        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{full_path}"

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.cache"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    logger.debug(f"Using cached document: {doc_path}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {doc_path}: {e}")

        try:
            file_ext = Path(doc_path).suffix.lower()

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()

                if file_ext in image_extensions:
                    image_data = response.content
                    file_size = len(image_data)

                    if file_size > max_image_size:
                        content: dict[str, Any] = {
                            "type": "image",
                            "size": file_size,
                            "size_exceeded": True,
                            "url": url,
                            "fetched": True,
                        }
                    else:
                        base64_content = base64.b64encode(image_data).decode("utf-8")
                        content = {
                            "type": "image",
                            "content": base64_content,
                            "size": file_size,
                            "url": url,
                            "fetched": True,
                        }
                elif file_ext in text_extensions:
                    text_content = response.text
                    content = {
                        "type": "text",
                        "content": text_content,
                        "size": len(text_content),
                        "fetched": True,
                    }
                else:
                    return None

                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(content, f)
                    logger.debug(f"Cached document: {doc_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache document {doc_path}: {e}")

                return content

        except Exception as e:
            logger.error(f"Failed to fetch document {doc_path} from {url}: {e}")
            return None

    return fetch_document


# === Local Loading ===


def load_from_local(
    path: str, config: Config | dict[str, Any] | None = None
) -> Result[list[Skill], LoadError]:
    """Load skills from a local directory.

    Parameters
    ----------
    path : str
        Path to local directory containing skills.
    config : Config | dict[str, Any] | None
        Configuration with document loading settings.

    Returns
    -------
    Result[list[Skill], LoadError]
        Ok with list of loaded skills or Err with LoadError.
    """
    skills: list[Skill] = []

    # Handle both Config objects and dicts for backward compatibility
    if config is None:
        config_dict: dict[str, Any] = {}
    elif isinstance(config, Config):
        config_dict = config_to_dict(config)
    else:
        config_dict = config

    load_documents = config_dict.get("load_skill_documents", True)
    text_extensions = config_dict.get(
        "text_file_extensions",
        [".md", ".py", ".txt", ".json", ".yaml", ".yml", ".sh", ".r", ".ipynb"],
    )
    image_extensions = config_dict.get(
        "allowed_image_extensions", [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]
    )
    max_image_size = config_dict.get("max_image_size_bytes", 5242880)

    try:
        local_path = Path(path).expanduser().resolve()

        if not local_path.exists():
            logger.warning(f"Local path {path} does not exist, skipping")
            return Ok([])

        if not local_path.is_dir():
            logger.warning(f"Local path {path} is not a directory, skipping")
            return Ok([])

        skill_files = list(local_path.rglob("SKILL.md"))

        for skill_file in skill_files:
            try:
                content = skill_file.read_text(encoding="utf-8")
                result = parse_skill_md(content, str(skill_file))

                if result.is_ok():
                    skill = result.value
                    if load_documents:
                        skill_dir = skill_file.parent
                        documents = _load_documents_from_directory(
                            skill_dir, text_extensions, image_extensions, max_image_size
                        )
                        skill.documents = documents
                        if documents:
                            logger.info(
                                f"Loaded {len(documents)} additional documents "
                                f"for skill: {skill.name}"
                            )

                    skills.append(skill)
                    logger.info(f"Loaded skill: {skill.name} from {skill_file}")
                else:
                    logger.warning(f"Failed to parse {skill_file}: {result.error.message}")

            except Exception as e:
                logger.error(f"Error reading {skill_file}: {e}")
                continue

        logger.info(f"Loaded {len(skills)} skills from local path {path}")
        return Ok(skills)

    except Exception as e:
        return Err(
            LoadError(
                message=f"Error accessing local path: {e}",
                source=path,
                cause=str(e),
            )
        )


# === GitHub Loading ===


def _parse_github_url(url: str) -> tuple[str, str, str, str]:
    """Parse GitHub URL to extract owner, repo, branch, and subpath.

    Returns
    -------
    tuple[str, str, str, str]
        (owner, repo, branch, subpath)
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")

    owner = path_parts[0]
    repo = path_parts[1]
    branch = "main"
    subpath = ""

    if len(path_parts) > 3 and path_parts[2] == "tree":
        branch = path_parts[3]
        if len(path_parts) > 4:
            subpath = "/".join(path_parts[4:])

    return owner, repo, branch, subpath


def _load_skills_from_tree(
    owner: str,
    repo: str,
    branch: str,
    subpath: str,
    tree_data: dict[str, Any],
    url: str,
    config_dict: dict[str, Any],
) -> list[Skill]:
    """Load skills from a GitHub tree."""
    skills: list[Skill] = []

    load_documents = config_dict.get("load_skill_documents", True)
    text_extensions = config_dict.get(
        "text_file_extensions",
        [".md", ".py", ".txt", ".json", ".yaml", ".yml", ".sh", ".r", ".ipynb"],
    )
    image_extensions = config_dict.get(
        "allowed_image_extensions", [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]
    )
    max_image_size = config_dict.get("max_image_size_bytes", 5242880)

    skill_paths = []
    for item in tree_data.get("tree", []):
        if item["type"] == "blob" and item["path"].endswith("SKILL.md"):
            if subpath:
                if item["path"].startswith(subpath):
                    skill_paths.append(item["path"])
            else:
                skill_paths.append(item["path"])

    for skill_path in skill_paths:
        try:
            raw_url = (
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{skill_path}"
            )

            with httpx.Client(timeout=30.0) as client:
                response = client.get(raw_url)
                response.raise_for_status()
                content = response.text

            source = f"{url}/tree/{branch}/{skill_path}"
            result = parse_skill_md(content, source)

            if result.is_ok():
                skill = result.value
                if load_documents:
                    skill_dir_path = str(Path(skill_path).parent)
                    if skill_dir_path == ".":
                        skill_dir_path = ""

                    documents = _get_document_metadata_from_github(
                        owner,
                        repo,
                        branch,
                        skill_dir_path,
                        tree_data,
                        text_extensions,
                        image_extensions,
                    )

                    fetcher = _create_document_fetcher(
                        owner,
                        repo,
                        branch,
                        skill_dir_path,
                        text_extensions,
                        image_extensions,
                        max_image_size,
                    )

                    skill.documents = documents
                    skill._document_fetcher = fetcher

                    if documents:
                        logger.info(
                            f"Found {len(documents)} additional documents "
                            f"for skill: {skill.name}"
                        )

                skills.append(skill)
                logger.info(f"Loaded skill: {skill.name} from {source}")
            else:
                logger.warning(
                    f"Failed to parse {skill_path}: {result.error.message}"
                )

        except Exception as e:
            logger.error(f"Error loading {skill_path} from GitHub: {e}")
            continue

    return skills


def load_from_github(
    url: str, subpath: str = "", config: Config | dict[str, Any] | None = None
) -> Result[list[Skill], LoadError]:
    """Load skills from a GitHub repository.

    Parameters
    ----------
    url : str
        GitHub repository URL.
    subpath : str, optional
        Subdirectory within the repo to search.
    config : Config | dict[str, Any] | None
        Configuration with document loading settings.

    Returns
    -------
    Result[list[Skill], LoadError]
        Ok with list of loaded skills or Err with LoadError.
    """
    # Handle both Config objects and dicts
    if config is None:
        config_dict: dict[str, Any] = {}
    elif isinstance(config, Config):
        config_dict = config_to_dict(config)
    else:
        config_dict = config

    try:
        owner, repo, branch, url_subpath = _parse_github_url(url)
        if not subpath and url_subpath:
            subpath = url_subpath
            logger.info(f"Extracted subpath from URL: {subpath}")

        if subpath:
            logger.info(
                f"Loading skills from GitHub: {owner}/{repo} "
                f"(branch: {branch}, subpath: {subpath})"
            )
        else:
            logger.info(
                f"Loading skills from GitHub: {owner}/{repo} (branch: {branch})"
            )

        cache_path = _get_cache_path(url, branch)
        tree_data = _load_from_cache(cache_path)

        if tree_data is None:
            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
                "?recursive=1"
            )

            with httpx.Client(timeout=30.0) as client:
                response = client.get(api_url)
                response.raise_for_status()
                tree_data = response.json()

            _save_to_cache(cache_path, tree_data)

        skills = _load_skills_from_tree(
            owner, repo, branch, subpath, tree_data, url, config_dict
        )

        logger.info(f"Loaded {len(skills)} skills from GitHub repo {url}")
        return Ok(skills)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info(
                f"Branch 'main' not found, trying 'master' for {url}"
            )
            try:
                owner, repo, _, _ = _parse_github_url(url)
                branch = "master"

                cache_path = _get_cache_path(url, branch)
                tree_data = _load_from_cache(cache_path)

                if tree_data is None:
                    api_url = (
                        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
                        "?recursive=1"
                    )

                    with httpx.Client(timeout=30.0) as client:
                        response = client.get(api_url)
                        response.raise_for_status()
                        tree_data = response.json()

                    _save_to_cache(cache_path, tree_data)

                skills = _load_skills_from_tree(
                    owner, repo, branch, subpath, tree_data, url, config_dict
                )

                logger.info(f"Loaded {len(skills)} skills from GitHub repo {url}")
                return Ok(skills)

            except Exception as e2:
                return Err(
                    LoadError(
                        message=f"Failed to load from GitHub (tried main and master): {e2}",
                        source=url,
                        cause=str(e2),
                    )
                )
        else:
            return Err(
                LoadError(
                    message=f"HTTP error loading from GitHub: {e}",
                    source=url,
                    cause=str(e),
                )
            )

    except Exception as e:
        return Err(
            LoadError(
                message=f"Error loading from GitHub: {e}",
                source=url,
                cause=str(e),
            )
        )


# === Batch Loading ===


def load_all_skills(
    skill_sources: list[dict[str, Any]], config: Config | dict[str, Any] | None = None
) -> list[Skill]:
    """Load skills from all configured sources.

    Parameters
    ----------
    skill_sources : list[dict[str, Any]]
        List of skill source configurations.
    config : Config | dict[str, Any] | None
        Configuration with document loading settings.

    Returns
    -------
    list[Skill]
        All loaded skills from all sources.
    """
    all_skills: list[Skill] = []

    for source_config in skill_sources:
        source_type = source_config.get("type")

        if source_type == "github":
            url = source_config.get("url")
            subpath = source_config.get("subpath", "")
            if url:
                result = load_from_github(url, subpath, config)
                if result.is_ok():
                    all_skills.extend(result.value)
                else:
                    logger.error(f"Failed to load from {url}: {result.error.message}")

        elif source_type == "local":
            path = source_config.get("path")
            if path:
                result = load_from_local(path, config)
                if result.is_ok():
                    all_skills.extend(result.value)
                else:
                    logger.error(f"Failed to load from {path}: {result.error.message}")

        else:
            logger.warning(f"Unknown source type: {source_type}")

    logger.info(f"Total skills loaded: {len(all_skills)}")
    return all_skills


def load_skills_in_batches(
    skill_sources: list[dict[str, Any]],
    config: Config | dict[str, Any] | None,
    batch_callback: Callable[[list[Skill], int], None],
    batch_size: int = 10,
) -> None:
    """Load skills from all sources in batches with callbacks.

    Parameters
    ----------
    skill_sources : list[dict[str, Any]]
        List of skill source configurations.
    config : Config | dict[str, Any] | None
        Configuration with document loading settings.
    batch_callback : Callable[[list[Skill], int], None]
        Callback called with (batch_skills, total_loaded) after each batch.
    batch_size : int, optional
        Number of skills per batch, by default 10.
    """
    current_batch: list[Skill] = []
    total_loaded = 0

    def process_batch() -> None:
        """Process and clear the current batch."""
        nonlocal total_loaded
        if current_batch:
            total_loaded += len(current_batch)
            batch_callback(current_batch.copy(), total_loaded)
            current_batch.clear()

    for source_config in skill_sources:
        source_type = source_config.get("type")

        try:
            if source_type == "github":
                url = source_config.get("url")
                subpath = source_config.get("subpath", "")
                if url:
                    result = load_from_github(url, subpath, config)
                    if result.is_ok():
                        for skill in result.value:
                            current_batch.append(skill)
                            if len(current_batch) >= batch_size:
                                process_batch()
                    else:
                        logger.error(
                            f"Failed to load from {url}: {result.error.message}"
                        )

            elif source_type == "local":
                path = source_config.get("path")
                if path:
                    result = load_from_local(path, config)
                    if result.is_ok():
                        for skill in result.value:
                            current_batch.append(skill)
                            if len(current_batch) >= batch_size:
                                process_batch()
                    else:
                        logger.error(
                            f"Failed to load from {path}: {result.error.message}"
                        )

            else:
                logger.warning(f"Unknown source type: {source_type}")

        except Exception as e:
            logger.error(f"Error loading from source {source_config}: {e}")
            continue

    process_batch()

    logger.info(f"Finished loading {total_loaded} skills in batches")

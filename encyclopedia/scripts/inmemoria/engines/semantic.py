"""Semantic analysis engine for extracting concepts from code."""

from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable

from ..config import get_config
from ..storage.database import Database
from ..types import (
    SemanticConcept,
    CodebaseAnalysisResult,
    ComplexityMetrics,
    LineRange,
)

logger = logging.getLogger(__name__)


class SemanticEngine:
    """Engine for semantic code analysis.

    Extracts semantic concepts (classes, functions, interfaces) from code
    using regex-based parsing. For production use, consider integrating
    tree-sitter for more accurate AST-based parsing.
    """

    def __init__(self, database: Database):
        """Initialize semantic engine.

        Args:
            database: Database instance for persistence.
        """
        self._db = database
        self._config = get_config()
        self._cache: dict[str, tuple[list[dict[str, Any]], float]] = {}

    def analyze_codebase(
        self,
        path: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> CodebaseAnalysisResult:
        """Analyze a codebase to extract high-level information.

        Args:
            path: Path to the codebase.
            progress_callback: Optional callback for progress updates.

        Returns:
            CodebaseAnalysisResult with languages, frameworks, complexity, etc.
        """
        project_path = Path(path).resolve()
        if not project_path.exists():
            return CodebaseAnalysisResult(
                languages=["unknown"],
                frameworks=[],
                complexity=ComplexityMetrics(cyclomatic=0, cognitive=0, lines=0),
                concepts=[],
                analysis_status="degraded",
                errors=[f"Path does not exist: {path}"],
            )

        # Collect files
        files = self._collect_source_files(project_path)
        if progress_callback:
            progress_callback(0, len(files), "Analyzing codebase...")

        # Detect languages
        languages = self._detect_languages(files)

        # Detect frameworks
        frameworks = self._detect_frameworks(project_path, languages)

        # Calculate complexity
        complexity = self._calculate_complexity(files, progress_callback)

        # Extract concepts (sample)
        concepts = self._extract_concepts_sample(files[:100])  # Limit for performance

        # Detect entry points
        entry_points = self._detect_entry_points(project_path, frameworks)

        # Map key directories
        key_directories = self._map_key_directories(project_path)

        return CodebaseAnalysisResult(
            languages=languages,
            frameworks=frameworks,
            complexity=complexity,
            concepts=concepts,
            entry_points=entry_points,
            key_directories=key_directories,
        )

    def analyze_file_content(
        self, file_path: str, content: str
    ) -> list[dict[str, Any]]:
        """Analyze file content and extract concepts.

        Args:
            file_path: Path to the file.
            content: File content.

        Returns:
            List of extracted concepts.
        """
        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{file_path}:{content_hash}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            import time
            if time.time() - timestamp < self._config.cache_ttl_seconds:
                return cached

        language = self._config.detect_language(file_path)
        concepts = self._extract_concepts_from_content(file_path, content, language)

        # Cache result
        import time
        self._cache[cache_key] = (concepts, time.time())

        return concepts

    def learn_from_codebase(
        self,
        path: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Deep learning from codebase - extract and store all concepts.

        Args:
            path: Path to the codebase.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of all extracted concepts.
        """
        project_path = Path(path).resolve()
        files = self._collect_source_files(project_path)

        total = len(files)
        all_concepts: list[dict[str, Any]] = []

        for i, file_path in enumerate(files):
            if progress_callback and i % 10 == 0:
                progress_callback(i, total, f"Analyzing {file_path.name}...")

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > self._config.max_file_size_bytes:
                    continue

                concepts = self.analyze_file_content(str(file_path), content)
                all_concepts.extend(concepts)

                # Store concepts in database
                for concept in concepts:
                    self._store_concept(concept)

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                continue

        if progress_callback:
            progress_callback(total, total, "Semantic analysis complete")

        return all_concepts

    def _collect_source_files(self, project_path: Path) -> list[Path]:
        """Collect all source files in project."""
        files: list[Path] = []

        for ext in self._config.SUPPORTED_EXTENSIONS:
            for file_path in project_path.rglob(f"*{ext}"):
                if not self._config.should_skip_path(str(file_path)):
                    files.append(file_path)

        return files

    def _detect_languages(self, files: list[Path]) -> list[str]:
        """Detect languages used in the project."""
        lang_counts: dict[str, int] = {}

        for file_path in files:
            lang = self._config.detect_language(str(file_path))
            if lang != "unknown":
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Sort by frequency
        sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
        return [lang for lang, _ in sorted_langs]

    def _detect_frameworks(self, project_path: Path, languages: list[str]) -> list[str]:
        """Detect frameworks used in the project."""
        frameworks: list[str] = []

        # Check package.json for JS/TS frameworks
        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                import json
                data = json.loads(package_json.read_text())
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}

                if "react" in deps:
                    frameworks.append("react")
                if "next" in deps:
                    frameworks.append("nextjs")
                if "svelte" in deps or "@sveltejs/kit" in deps:
                    frameworks.append("svelte")
                if "vue" in deps:
                    frameworks.append("vue")
                if "express" in deps:
                    frameworks.append("express")
                if "fastify" in deps:
                    frameworks.append("fastify")
            except Exception:
                pass

        # Check for Python frameworks
        requirements = project_path / "requirements.txt"
        pyproject = project_path / "pyproject.toml"

        python_deps = ""
        if requirements.exists():
            python_deps = requirements.read_text().lower()
        if pyproject.exists():
            python_deps += pyproject.read_text().lower()

        if "fastapi" in python_deps:
            frameworks.append("fastapi")
        if "django" in python_deps:
            frameworks.append("django")
        if "flask" in python_deps:
            frameworks.append("flask")

        # Check Cargo.toml for Rust frameworks
        cargo = project_path / "Cargo.toml"
        if cargo.exists():
            cargo_content = cargo.read_text().lower()
            if "actix" in cargo_content:
                frameworks.append("actix")
            if "axum" in cargo_content:
                frameworks.append("axum")
            if "rocket" in cargo_content:
                frameworks.append("rocket")

        return frameworks

    def _calculate_complexity(
        self,
        files: list[Path],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ComplexityMetrics:
        """Calculate aggregate complexity metrics."""
        total_lines = 0
        total_cyclomatic = 0
        total_cognitive = 0
        file_count = 0

        for i, file_path in enumerate(files[:500]):  # Limit for performance
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = len(content.splitlines())
                total_lines += lines

                # Simple cyclomatic complexity estimation
                cyclomatic = self._estimate_cyclomatic(content)
                total_cyclomatic += cyclomatic

                # Simple cognitive complexity estimation
                cognitive = self._estimate_cognitive(content)
                total_cognitive += cognitive

                file_count += 1

            except Exception:
                continue

        avg_cyclomatic = total_cyclomatic / max(file_count, 1)
        avg_cognitive = total_cognitive / max(file_count, 1)

        return ComplexityMetrics(
            cyclomatic=avg_cyclomatic,
            cognitive=avg_cognitive,
            lines=total_lines,
        )

    def _estimate_cyclomatic(self, content: str) -> float:
        """Estimate cyclomatic complexity from code."""
        # Count decision points
        patterns = [
            r"\bif\b",
            r"\belif\b",
            r"\belse\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\b\?\b",  # Ternary
            r"\b&&\b",
            r"\b\|\|\b",
        ]

        count = 1  # Base complexity
        for pattern in patterns:
            count += len(re.findall(pattern, content))

        return count

    def _estimate_cognitive(self, content: str) -> float:
        """Estimate cognitive complexity from code."""
        # Count nesting and control flow
        lines = content.splitlines()
        cognitive = 0
        nesting = 0

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Increment for control structures
            if re.search(r"\b(if|for|while|switch|try)\b", stripped):
                cognitive += 1 + nesting
                nesting += 1
            elif re.search(r"\b(else|elif|catch|finally)\b", stripped):
                cognitive += 1
            elif stripped.startswith("}") or stripped == "":
                nesting = max(0, nesting - 1)

        return cognitive

    def _extract_concepts_sample(self, files: list[Path]) -> list[dict[str, Any]]:
        """Extract a sample of concepts for quick analysis."""
        concepts: list[dict[str, Any]] = []

        for file_path in files[:50]:  # Sample
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                language = self._config.detect_language(str(file_path))
                file_concepts = self._extract_concepts_from_content(
                    str(file_path), content, language
                )
                concepts.extend(file_concepts[:10])  # Limit per file
            except Exception:
                continue

        return concepts

    def _extract_concepts_from_content(
        self, file_path: str, content: str, language: str
    ) -> list[dict[str, Any]]:
        """Extract concepts from file content using regex patterns."""
        concepts: list[dict[str, Any]] = []
        lines = content.splitlines()

        # Language-specific patterns
        patterns = self._get_language_patterns(language)

        for i, line in enumerate(lines):
            for pattern_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if match.groups() else match.group(0)
                    concepts.append({
                        "id": hashlib.md5(f"{file_path}:{i}:{name}".encode()).hexdigest()[:16],
                        "name": name,
                        "type": pattern_name,
                        "confidence": 0.7,
                        "file_path": file_path,
                        "line_range": {"start": i + 1, "end": i + 1},
                        "relationships": {},
                    })

        return concepts

    def _get_language_patterns(self, language: str) -> dict[str, str]:
        """Get regex patterns for concept extraction by language."""
        common = {
            "class": r"class\s+(\w+)",
            "function": r"function\s+(\w+)",
        }

        patterns: dict[str, dict[str, str]] = {
            "typescript": {
                **common,
                "interface": r"interface\s+(\w+)",
                "type": r"type\s+(\w+)",
                "const_function": r"const\s+(\w+)\s*=\s*(?:async\s*)?\(",
            },
            "javascript": {
                **common,
                "const_function": r"const\s+(\w+)\s*=\s*(?:async\s*)?\(",
            },
            "python": {
                "class": r"class\s+(\w+)",
                "function": r"def\s+(\w+)",
                "async_function": r"async\s+def\s+(\w+)",
            },
            "rust": {
                "struct": r"struct\s+(\w+)",
                "enum": r"enum\s+(\w+)",
                "impl": r"impl\s+(\w+)",
                "function": r"fn\s+(\w+)",
                "trait": r"trait\s+(\w+)",
            },
            "go": {
                "struct": r"type\s+(\w+)\s+struct",
                "interface": r"type\s+(\w+)\s+interface",
                "function": r"func\s+(\w+)",
            },
        }

        return patterns.get(language, common)

    def _detect_entry_points(
        self, project_path: Path, frameworks: list[str]
    ) -> list[dict[str, Any]]:
        """Detect project entry points."""
        entry_points: list[dict[str, Any]] = []

        # Common entry point patterns
        patterns = [
            # React/Next.js
            ("src/index.tsx", "web", "react"),
            ("src/index.jsx", "web", "react"),
            ("src/App.tsx", "web", "react"),
            ("pages/_app.tsx", "web", "nextjs"),
            ("app/layout.tsx", "web", "nextjs"),
            # Svelte
            ("src/routes/+page.svelte", "web", "svelte"),
            ("src/main.ts", "web", "svelte"),
            # Node/Express
            ("server.js", "api", "express"),
            ("src/server.ts", "api", "express"),
            ("src/index.ts", "api", "express"),
            ("app.js", "api", "express"),
            # Python
            ("main.py", "api", "fastapi"),
            ("app.py", "api", "flask"),
            ("manage.py", "cli", "django"),
            # CLI
            ("cli.js", "cli", None),
            ("src/cli.ts", "cli", None),
        ]

        for rel_path, entry_type, framework in patterns:
            full_path = project_path / rel_path
            if full_path.exists():
                entry_points.append({
                    "type": entry_type,
                    "filePath": rel_path,
                    "framework": framework,
                })

        return entry_points

    def _map_key_directories(self, project_path: Path) -> list[dict[str, Any]]:
        """Map key directories in the project."""
        key_dirs: list[dict[str, Any]] = []

        patterns = [
            ("src/components", "components"),
            ("components", "components"),
            ("src/utils", "utils"),
            ("utils", "utils"),
            ("lib", "library"),
            ("src/lib", "library"),
            ("src/services", "services"),
            ("services", "services"),
            ("src/api", "api"),
            ("api", "api"),
            ("routes", "routes"),
            ("src/routes", "routes"),
            ("middleware", "middleware"),
            ("src/models", "models"),
            ("models", "models"),
            ("tests", "tests"),
            ("__tests__", "tests"),
        ]

        for rel_path, dir_type in patterns:
            full_path = project_path / rel_path
            if full_path.exists() and full_path.is_dir():
                file_count = sum(1 for _ in full_path.rglob("*") if _.is_file())
                key_dirs.append({
                    "path": rel_path,
                    "type": dir_type,
                    "fileCount": file_count,
                })

        return key_dirs

    def _store_concept(self, concept: dict[str, Any]) -> None:
        """Store concept in database."""
        line_range = concept.get("line_range", {"start": 0, "end": 0})
        self._db.insert_semantic_concept(
            SemanticConcept(
                id=concept["id"],
                concept_name=concept["name"],
                concept_type=concept["type"],
                confidence_score=concept.get("confidence", 0.5),
                relationships=concept.get("relationships", {}),
                evolution_history={},
                file_path=concept["file_path"],
                line_range=LineRange(
                    start=line_range["start"],
                    end=line_range["end"],
                ),
            )
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        self._cache.clear()

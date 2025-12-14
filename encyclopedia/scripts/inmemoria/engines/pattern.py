"""Pattern analysis engine for learning coding patterns."""

from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable

from ..config import get_config
from ..storage.database import Database
from ..types import (
    DeveloperPattern,
    PatternAnalysisResult,
    RelevantPattern,
    ApproachPrediction,
    Complexity,
    FeatureMap,
    FileRouting,
    WorkType,
)

logger = logging.getLogger(__name__)


class PatternEngine:
    """Engine for learning and analyzing coding patterns."""

    def __init__(self, database: Database):
        """Initialize pattern engine.

        Args:
            database: Database instance for persistence.
        """
        self._db = database
        self._config = get_config()

    def extract_patterns(self, path: str) -> list[dict[str, Any]]:
        """Extract patterns from a codebase path.

        Args:
            path: Path to the codebase.

        Returns:
            List of extracted patterns.
        """
        project_path = Path(path).resolve()
        patterns: list[dict[str, Any]] = []

        for ext in self._config.SUPPORTED_EXTENSIONS:
            for file_path in project_path.rglob(f"*{ext}"):
                if self._config.should_skip_path(str(file_path)):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    file_patterns = self._extract_patterns_from_content(content, str(file_path))
                    patterns.extend(file_patterns)
                except Exception as e:
                    logger.warning(f"Failed to extract patterns from {file_path}: {e}")

        return self._deduplicate_patterns(patterns)

    def analyze_file_patterns(
        self, file_path: str, content: str
    ) -> list[dict[str, Any]]:
        """Analyze patterns in a specific file.

        Args:
            file_path: Path to the file.
            content: File content.

        Returns:
            List of detected patterns with descriptions.
        """
        patterns = self._extract_patterns_from_content(content, file_path)
        return [
            {
                "type": p["type"],
                "description": p.get("description", f"{p['type']} pattern detected"),
                "confidence": p.get("confidence", 0.7),
            }
            for p in patterns
        ]

    def learn_from_codebase(
        self,
        path: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Learn patterns from codebase and store them.

        Args:
            path: Path to the codebase.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of learned patterns.
        """
        if progress_callback:
            progress_callback(0, 100, "Starting pattern analysis...")

        project_path = Path(path).resolve()
        all_patterns: dict[str, dict[str, Any]] = {}

        files = list(project_path.rglob("*"))
        source_files = [
            f for f in files
            if f.is_file()
            and f.suffix.lower() in self._config.SUPPORTED_EXTENSIONS
            and not self._config.should_skip_path(str(f))
        ]

        total = len(source_files)
        for i, file_path in enumerate(source_files):
            if progress_callback and i % 20 == 0:
                progress_callback(
                    int((i / total) * 80), 100, f"Analyzing patterns... ({i}/{total})"
                )

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > self._config.max_file_size_bytes:
                    continue

                file_patterns = self._extract_patterns_from_content(content, str(file_path))
                for pattern in file_patterns:
                    key = pattern["type"]
                    if key in all_patterns:
                        all_patterns[key]["frequency"] += 1
                        all_patterns[key]["contexts"].add(
                            self._config.detect_language(str(file_path))
                        )
                        if pattern.get("example"):
                            all_patterns[key]["examples"].append(pattern["example"])
                    else:
                        all_patterns[key] = {
                            "id": hashlib.md5(key.encode()).hexdigest()[:16],
                            "type": key,
                            "content": {"description": pattern.get("description", key)},
                            "frequency": 1,
                            "confidence": pattern.get("confidence", 0.5),
                            "contexts": {self._config.detect_language(str(file_path))},
                            "examples": [pattern["example"]] if pattern.get("example") else [],
                        }
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

        if progress_callback:
            progress_callback(80, 100, "Storing patterns...")

        # Convert and store patterns
        result: list[dict[str, Any]] = []
        for pattern_data in all_patterns.values():
            pattern = DeveloperPattern(
                pattern_id=pattern_data["id"],
                pattern_type=pattern_data["type"],
                pattern_content=pattern_data["content"],
                frequency=pattern_data["frequency"],
                contexts=list(pattern_data["contexts"]),
                examples=[{"code": ex} for ex in pattern_data["examples"][:5]],
                confidence=min(0.9, 0.3 + (pattern_data["frequency"] * 0.05)),
            )
            self._db.insert_developer_pattern(pattern)
            result.append({
                "id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "content": pattern.pattern_content,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "contexts": pattern.contexts,
                "examples": pattern.examples,
            })

        if progress_callback:
            progress_callback(100, 100, f"Learned {len(result)} patterns")

        return result

    def analyze_file_change(self, change: dict[str, Any]) -> PatternAnalysisResult:
        """Analyze a file change for patterns.

        Args:
            change: Dictionary with type, path, content, language keys.

        Returns:
            PatternAnalysisResult with detected patterns and recommendations.
        """
        content = change.get("content", "")
        file_path = change.get("path", "")

        detected: list[str] = []
        violations: list[str] = []
        recommendations: list[str] = []

        # Check naming conventions
        if re.search(r"function\s+[a-z][a-zA-Z]*", content):
            detected.append("camelCase_function_naming")

        if re.search(r"class\s+[A-Z][a-zA-Z]*", content):
            detected.append("PascalCase_class_naming")

        # Check for violations
        if re.search(r"function\s+[A-Z]", content):
            violations.append("function_naming_violation")
            recommendations.append("Use camelCase for function names")

        if re.search(r"class\s+[a-z]", content):
            violations.append("class_naming_violation")
            recommendations.append("Use PascalCase for class names")

        # Check for testing patterns
        if re.search(r"describe|it|test|expect", content):
            detected.append("testing")

        # Check for API patterns
        if re.search(r"app\.(get|post|put|delete)|router\.(get|post|put|delete)", content):
            detected.append("api_design")

        return PatternAnalysisResult(
            detected=detected,
            violations=violations,
            recommendations=recommendations,
        )

    def find_relevant_patterns(
        self,
        problem_description: str,
        current_file: str | None = None,
        selected_code: str | None = None,
    ) -> list[RelevantPattern]:
        """Find patterns relevant to a problem description.

        Args:
            problem_description: Description of the problem/task.
            current_file: Optional current file path.
            selected_code: Optional selected code snippet.

        Returns:
            List of relevant patterns with confidence scores.
        """
        # Get all patterns from database
        db_patterns = self._db.get_developer_patterns(limit=200)
        if not db_patterns:
            return []

        # Extract keywords from problem
        keywords = self._extract_keywords(problem_description.lower())

        # Score each pattern
        scored: list[tuple[DeveloperPattern, float]] = []
        for pattern in db_patterns:
            score = 0.0
            pattern_content = str(pattern.pattern_content).lower()
            pattern_type = pattern.pattern_type.lower()

            for keyword in keywords:
                if keyword in pattern_type:
                    score += 0.3
                if keyword in pattern_content:
                    score += 0.2

            score += pattern.confidence * 0.3
            score += min(pattern.frequency / 10, 1.0) * 0.2

            if score > 0.3:
                scored.append((pattern, score))

        # Sort by score and return top 10
        scored.sort(key=lambda x: -x[1])
        return [
            RelevantPattern(
                pattern_id=p.pattern_id,
                pattern_type=p.pattern_type,
                pattern_content=p.pattern_content,
                frequency=p.frequency,
                contexts=p.contexts,
                examples=[{"code": ex.get("code", "")} for ex in p.examples],
                confidence=p.confidence,
            )
            for p, _ in scored[:10]
        ]

    def predict_approach(
        self,
        problem_description: str,
        context: dict[str, Any] | None = None,
    ) -> ApproachPrediction:
        """Predict an approach for solving a problem.

        Args:
            problem_description: Description of the problem.
            context: Optional context information.

        Returns:
            ApproachPrediction with suggested approach.
        """
        # Analyze problem description
        lower_desc = problem_description.lower()

        # Determine complexity
        word_count = len(problem_description.split())
        if word_count < 10:
            complexity = Complexity.LOW
        elif word_count < 25:
            complexity = Complexity.MEDIUM
        else:
            complexity = Complexity.HIGH

        # Find relevant patterns
        relevant_patterns = self.find_relevant_patterns(problem_description)
        pattern_names = [p.pattern_type for p in relevant_patterns[:5]]

        # Generate approach based on keywords
        if "test" in lower_desc:
            approach = "Test-driven development"
            reasoning = "Testing keywords detected in problem description"
        elif "api" in lower_desc or "endpoint" in lower_desc:
            approach = "API-first design"
            reasoning = "API-related keywords detected"
        elif "refactor" in lower_desc:
            approach = "Incremental refactoring with tests"
            reasoning = "Refactoring task detected"
        elif "fix" in lower_desc or "bug" in lower_desc:
            approach = "Debug and fix with regression tests"
            reasoning = "Bug fix task detected"
        else:
            approach = "Iterative development with testing"
            reasoning = "General development approach based on complexity"

        return ApproachPrediction(
            approach=approach,
            confidence=0.6 if relevant_patterns else 0.4,
            reasoning=reasoning,
            patterns=pattern_names,
            complexity=complexity,
        )

    def build_feature_map(self, project_path: str) -> list[dict[str, Any]]:
        """Build feature-to-file mapping for a project.

        Args:
            project_path: Path to the project.

        Returns:
            List of feature maps.
        """
        path = Path(project_path).resolve()
        feature_maps: list[dict[str, Any]] = []

        # Feature patterns to look for
        feature_patterns = {
            "authentication": ["auth", "authentication", "login", "signup", "register"],
            "api": ["api", "routes", "endpoints", "controllers"],
            "database": ["db", "database", "models", "schemas", "migrations"],
            "ui-components": ["components", "ui"],
            "views": ["views", "pages", "screens"],
            "services": ["services", "api-clients"],
            "utilities": ["utils", "helpers", "lib"],
            "testing": ["tests", "__tests__", "test"],
            "configuration": ["config", ".config", "settings"],
            "middleware": ["middleware", "middlewares"],
        }

        for feature_name, dir_patterns in feature_patterns.items():
            primary_files: list[str] = []
            related_files: list[str] = []

            for dir_pattern in dir_patterns:
                # Check src/pattern and pattern
                for prefix in ["src", ""]:
                    check_path = path / prefix / dir_pattern if prefix else path / dir_pattern
                    if check_path.exists() and check_path.is_dir():
                        files = self._collect_files_in_directory(check_path, path)
                        if files:
                            mid = len(files) // 2
                            primary_files.extend(files[:mid] if mid > 0 else files)
                            related_files.extend(files[mid:] if mid > 0 else [])

            if primary_files:
                feature_maps.append({
                    "id": hashlib.md5(f"{project_path}:{feature_name}".encode()).hexdigest()[:16],
                    "featureName": feature_name,
                    "primaryFiles": list(set(primary_files)),
                    "relatedFiles": list(set(related_files)),
                    "dependencies": [],
                })

        # Store feature maps
        for fm in feature_maps:
            self._db.insert_feature_map(
                FeatureMap(
                    id=fm["id"],
                    project_path=project_path,
                    feature_name=fm["featureName"],
                    primary_files=fm["primaryFiles"],
                    related_files=fm["relatedFiles"],
                    dependencies=fm["dependencies"],
                )
            )

        return feature_maps

    def route_request_to_files(
        self, problem_description: str, project_path: str
    ) -> FileRouting | None:
        """Route a request to relevant files.

        Args:
            problem_description: Description of the task.
            project_path: Path to the project.

        Returns:
            FileRouting with target files, or None if no match.
        """
        feature_maps = self._db.get_feature_maps(project_path)
        if not feature_maps:
            return None

        lower_desc = problem_description.lower()

        # Determine work type
        if "fix" in lower_desc or "bug" in lower_desc:
            work_type = WorkType.BUGFIX
        elif "refactor" in lower_desc or "improve" in lower_desc:
            work_type = WorkType.REFACTOR
        elif "test" in lower_desc:
            work_type = WorkType.TEST
        else:
            work_type = WorkType.FEATURE

        # Keyword matching
        keyword_groups = {
            "authentication": ["auth", "login", "signup", "register", "password"],
            "api": ["api", "endpoint", "route", "controller", "rest"],
            "database": ["database", "db", "model", "schema", "migration", "query"],
            "ui-components": ["component", "ui", "button", "form", "input"],
            "views": ["view", "page", "screen", "template", "layout"],
            "services": ["service", "client", "provider", "manager"],
            "utilities": ["util", "helper", "function", "library"],
            "testing": ["test", "spec", "mock", "fixture"],
            "configuration": ["config", "setting", "environment"],
            "middleware": ["middleware", "interceptor", "guard"],
        }

        # Score features
        best_score = 0.0
        best_feature: FeatureMap | None = None
        best_keywords: list[str] = []

        for feature in feature_maps:
            keywords = keyword_groups.get(feature.feature_name, [])
            matched = [kw for kw in keywords if kw in lower_desc]
            score = len(matched) * 0.3

            # Direct name match boost
            if feature.feature_name.replace("-", " ") in lower_desc:
                score += 1.0
                matched.append(feature.feature_name)

            if score > best_score:
                best_score = score
                best_feature = feature
                best_keywords = matched

        if best_feature and best_score > 0:
            all_files = best_feature.primary_files + best_feature.related_files
            confidence = min(0.95, 0.3 + (best_score * 0.2))

            return FileRouting(
                intended_feature=best_feature.feature_name,
                target_files=all_files[:5],
                work_type=work_type,
                suggested_start_point=best_feature.primary_files[0] if best_feature.primary_files else all_files[0],
                confidence=confidence,
                reasoning=f"Matched keywords: {', '.join(best_keywords)}. Found {len(all_files)} relevant files.",
            )

        # Fallback with low confidence
        if feature_maps:
            first = feature_maps[0]
            all_files = first.primary_files + first.related_files
            return FileRouting(
                intended_feature=first.feature_name,
                target_files=all_files[:5],
                work_type=work_type,
                suggested_start_point=first.primary_files[0] if first.primary_files else "",
                confidence=0.2,
                reasoning="No keyword matches found. Suggesting most common feature as fallback.",
            )

        return None

    def get_pattern_statistics(self) -> dict[str, Any]:
        """Get statistics about learned patterns.

        Returns:
            Dictionary with pattern statistics.
        """
        all_patterns = self._db.get_developer_patterns(limit=100)

        by_type: dict[str, int] = {}
        for pattern in all_patterns:
            by_type[pattern.pattern_type] = by_type.get(pattern.pattern_type, 0) + 1

        most_used = sorted(all_patterns, key=lambda p: -p.frequency)[:10]
        recently_used = sorted(
            all_patterns,
            key=lambda p: p.last_seen or p.created_at or p.last_seen,
            reverse=True,
        )[:10]

        return {
            "total_patterns": len(all_patterns),
            "by_type": by_type,
            "most_used": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                }
                for p in most_used
            ],
            "recently_used": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "last_seen": p.last_seen.isoformat() if p.last_seen else None,
                }
                for p in recently_used
            ],
        }

    def _extract_patterns_from_content(
        self, content: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract patterns from file content."""
        patterns: list[dict[str, Any]] = []

        # Naming patterns
        if re.search(r"function\s+[a-z][a-zA-Z]*", content):
            patterns.append({
                "type": "camelCase_function_naming",
                "description": "Functions use camelCase naming convention",
                "confidence": 0.8,
            })

        if re.search(r"class\s+[A-Z][a-zA-Z]*", content):
            patterns.append({
                "type": "PascalCase_class_naming",
                "description": "Classes use PascalCase naming convention",
                "confidence": 0.8,
            })

        if re.search(r"[a-z]+_[a-z]+", content):
            patterns.append({
                "type": "snake_case_naming",
                "description": "Variables use snake_case naming convention",
                "confidence": 0.6,
            })

        # Testing patterns
        if re.search(r"describe|it|test|expect", content):
            patterns.append({
                "type": "testing",
                "description": "Testing pattern with describe/it/test blocks",
                "confidence": 0.9,
            })

        # API patterns
        if re.search(r"app\.(get|post|put|delete)|router\.(get|post|put|delete)", content):
            patterns.append({
                "type": "api_design",
                "description": "RESTful API design pattern",
                "confidence": 0.7,
            })

        # Dependency injection
        if re.search(r"constructor\([^)]*private|@Injectable", content):
            patterns.append({
                "type": "dependency_injection",
                "description": "Dependency injection pattern detected",
                "confidence": 0.8,
            })

        # Factory pattern
        if re.search(r"create\w+|factory|Factory", content):
            patterns.append({
                "type": "factory",
                "description": "Factory pattern for object creation",
                "confidence": 0.6,
            })

        # Singleton pattern
        if re.search(r"getInstance|_instance|singleton", content, re.IGNORECASE):
            patterns.append({
                "type": "singleton",
                "description": "Singleton pattern implementation",
                "confidence": 0.7,
            })

        # Async/await pattern
        if re.search(r"async\s+\w+|await\s+", content):
            patterns.append({
                "type": "async_await",
                "description": "Async/await pattern for asynchronous code",
                "confidence": 0.8,
            })

        # Error handling pattern
        if re.search(r"try\s*{|except\s+\w+|\.catch\(", content):
            patterns.append({
                "type": "error_handling",
                "description": "Error handling with try/catch/except",
                "confidence": 0.7,
            })

        return patterns

    def _deduplicate_patterns(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate patterns by type."""
        seen: dict[str, dict[str, Any]] = {}
        for pattern in patterns:
            key = pattern["type"]
            if key not in seen:
                seen[key] = pattern
            else:
                seen[key]["frequency"] = seen[key].get("frequency", 1) + 1

        return list(seen.values())

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "to", "from", "in", "on", "at", "by", "for", "with", "about",
            "as", "of", "and", "or", "but", "i", "we", "you", "it", "this", "that",
        }

        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return [w for w in words if len(w) > 2 and w not in stop_words]

    def _collect_files_in_directory(
        self, dir_path: Path, project_path: Path, max_depth: int = 5
    ) -> list[str]:
        """Collect source files in a directory."""
        files: list[str] = []

        try:
            for item in dir_path.rglob("*"):
                if item.is_file():
                    # Check depth
                    rel = item.relative_to(dir_path)
                    if len(rel.parts) > max_depth:
                        continue

                    # Check if source file
                    if item.suffix.lower() in self._config.SUPPORTED_EXTENSIONS:
                        if not self._config.should_skip_path(str(item)):
                            files.append(str(item.relative_to(project_path)))
        except Exception as e:
            logger.warning(f"Error collecting files from {dir_path}: {e}")

        return files

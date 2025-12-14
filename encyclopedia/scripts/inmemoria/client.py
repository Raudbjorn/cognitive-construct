"""Main client for inmemoria codebase intelligence library."""

from __future__ import annotations
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import get_config, Config
from .storage.database import Database
from .engines.semantic import SemanticEngine
from .engines.pattern import PatternEngine
from .types import (
    LearningResult,
    Blueprint,
    CodebaseAnalysisResult,
    RelevantPattern,
    ApproachPrediction,
    FileRouting,
    ProjectMetadata,
    SemanticConcept,
    DeveloperPattern,
    FeatureMap,
    EntryPoint,
    KeyDirectory,
)
from .result import Result, Ok, Err

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """Generate a unique ID."""
    import hashlib
    import random
    return hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:16]


class InMemoriaClient:
    """Client for inmemoria codebase intelligence.

    Usage:
        from inmemoria import InMemoriaClient

        client = InMemoriaClient("/path/to/project")

        # Learn from codebase (one-time setup)
        result = client.learn()
        if result.is_ok():
            print(f"Learned {result.value.concepts_learned} concepts")

        # Get project blueprint
        blueprint = client.get_blueprint()

        # Search for patterns
        patterns = client.find_patterns("authentication logic")

        # Predict approach for a task
        approach = client.predict_approach("add user registration")

        # Route task to files
        routing = client.route_to_files("fix login bug")
    """

    def __init__(
        self,
        project_path: str,
        config: Config | None = None,
        db_path: str | None = None,
    ):
        """Initialize InMemoria client.

        Args:
            project_path: Path to the project to analyze.
            config: Optional custom configuration.
            db_path: Optional custom database path.
        """
        self._project_path = str(Path(project_path).resolve())
        self._config = config or get_config()

        # Set up database
        if db_path:
            self._db_path = Path(db_path)
        else:
            self._db_path = self._config.get_database_path(self._project_path)

        self._db = Database(self._db_path)
        self._semantic_engine = SemanticEngine(self._db)
        self._pattern_engine = PatternEngine(self._db)

    @property
    def project_path(self) -> str:
        """Get the project path."""
        return self._project_path

    @property
    def database_path(self) -> Path:
        """Get the database path."""
        return self._db_path

    def close(self) -> None:
        """Close the client and release resources."""
        self._semantic_engine.cleanup()
        self._db.close()

    def __enter__(self) -> InMemoriaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def learn(
        self,
        force: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Result[LearningResult, str]:
        """Learn from the codebase.

        This is the main learning function that extracts semantic concepts,
        coding patterns, and builds feature maps.

        Args:
            force: If True, re-learn even if intelligence already exists.
            progress_callback: Optional callback for progress updates.

        Returns:
            Result containing LearningResult on success, error message on failure.
        """
        start_time = time.time()
        insights: list[str] = []

        try:
            # Check for existing intelligence
            if not force:
                existing = self._check_existing_intelligence()
                if existing and existing["concepts"] > 0:
                    return Ok(LearningResult(
                        success=True,
                        concepts_learned=existing["concepts"],
                        patterns_learned=existing["patterns"],
                        features_learned=existing["features"],
                        insights=["Using existing intelligence (use force=True to re-learn)"],
                        time_elapsed_ms=int((time.time() - start_time) * 1000),
                    ))

            # Phase 1: Codebase analysis
            insights.append("Phase 1: Analyzing codebase structure...")
            analysis = self._semantic_engine.analyze_codebase(
                self._project_path, progress_callback
            )
            insights.append(f"  Detected languages: {', '.join(analysis.languages)}")
            insights.append(f"  Found frameworks: {', '.join(analysis.frameworks) or 'none'}")
            if analysis.complexity:
                insights.append(
                    f"  Complexity: {analysis.complexity.cyclomatic:.1f} cyclomatic, "
                    f"{analysis.complexity.cognitive:.1f} cognitive"
                )

            # Phase 2: Semantic learning
            insights.append("Phase 2: Learning semantic concepts...")
            concepts = self._semantic_engine.learn_from_codebase(
                self._project_path, progress_callback
            )
            concept_types: dict[str, int] = {}
            for c in concepts:
                ctype = c.get("type", "unknown")
                concept_types[ctype] = concept_types.get(ctype, 0) + 1
            insights.append(f"  Extracted {len(concepts)} semantic concepts:")
            for ctype, count in concept_types.items():
                insights.append(f"    - {count} {ctype}{'s' if count > 1 else ''}")

            # Phase 3: Pattern learning
            insights.append("Phase 3: Discovering coding patterns...")
            patterns = self._pattern_engine.learn_from_codebase(
                self._project_path, progress_callback
            )
            pattern_types: dict[str, int] = {}
            for p in patterns:
                ptype = p.get("type", "unknown").split("_")[0]
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
            insights.append(f"  Identified {len(patterns)} coding patterns:")
            for ptype, count in pattern_types.items():
                insights.append(f"    - {count} {ptype} pattern{'s' if count > 1 else ''}")

            # Phase 4: Feature mapping
            insights.append("Phase 4: Building feature map...")
            feature_maps = self._pattern_engine.build_feature_map(self._project_path)
            total_files = sum(
                len(fm.get("primaryFiles", [])) + len(fm.get("relatedFiles", []))
                for fm in feature_maps
            )
            insights.append(f"  Mapped {len(feature_maps)} features to {total_files} files")

            # Phase 5: Store project metadata
            insights.append("Phase 5: Storing project metadata...")
            self._store_project_metadata(analysis)
            self._store_blueprint(analysis)

            elapsed_ms = int((time.time() - start_time) * 1000)
            insights.append(f"Learning completed in {elapsed_ms}ms")

            # Build blueprint
            blueprint = Blueprint(
                tech_stack=analysis.frameworks,
                entry_points={
                    ep["type"]: ep["filePath"]
                    for ep in analysis.entry_points
                },
                key_directories={
                    kd["type"]: kd["path"]
                    for kd in analysis.key_directories
                },
                architecture=self._infer_architecture(analysis),
            )

            return Ok(LearningResult(
                success=True,
                concepts_learned=len(concepts),
                patterns_learned=len(patterns),
                features_learned=len(feature_maps),
                insights=insights,
                time_elapsed_ms=elapsed_ms,
                blueprint=blueprint,
            ))

        except Exception as e:
            logger.exception("Learning failed")
            return Err(f"Learning failed: {e}")

    def analyze(
        self,
        path: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Result[CodebaseAnalysisResult, str]:
        """Analyze a codebase or subdirectory.

        Args:
            path: Optional path to analyze (defaults to project root).
            progress_callback: Optional callback for progress updates.

        Returns:
            Result containing CodebaseAnalysisResult on success.
        """
        try:
            target = path or self._project_path
            result = self._semantic_engine.analyze_codebase(target, progress_callback)
            return Ok(result)
        except Exception as e:
            logger.exception("Analysis failed")
            return Err(f"Analysis failed: {e}")

    def get_blueprint(self) -> Blueprint | None:
        """Get the project blueprint.

        Returns:
            Blueprint with tech stack, entry points, and key directories,
            or None if not learned yet.
        """
        metadata = self._db.get_project_metadata(self._project_path)
        if not metadata:
            return None

        entry_points = self._db.get_entry_points(self._project_path)
        key_dirs = self._db.get_key_directories(self._project_path)

        return Blueprint(
            tech_stack=metadata.framework_detected,
            entry_points={ep.entry_type: ep.file_path for ep in entry_points},
            key_directories={kd.directory_type: kd.directory_path for kd in key_dirs},
            architecture=self._infer_architecture_from_metadata(metadata, key_dirs),
        )

    def find_patterns(
        self,
        problem_description: str,
        current_file: str | None = None,
        selected_code: str | None = None,
    ) -> list[RelevantPattern]:
        """Find patterns relevant to a problem.

        Args:
            problem_description: Description of the problem/task.
            current_file: Optional current file path.
            selected_code: Optional selected code snippet.

        Returns:
            List of relevant patterns.
        """
        return self._pattern_engine.find_relevant_patterns(
            problem_description, current_file, selected_code
        )

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
        return self._pattern_engine.predict_approach(problem_description, context)

    def route_to_files(self, problem_description: str) -> FileRouting | None:
        """Route a task to relevant files.

        Args:
            problem_description: Description of the task.

        Returns:
            FileRouting with target files, or None if no match.
        """
        return self._pattern_engine.route_request_to_files(
            problem_description, self._project_path
        )

    def search_concepts(self, query: str, limit: int = 10) -> list[SemanticConcept]:
        """Search for semantic concepts.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching concepts.
        """
        # Simple text search in concepts
        all_concepts = self._db.get_semantic_concepts()
        query_lower = query.lower()

        matches = [
            c for c in all_concepts
            if query_lower in c.concept_name.lower()
            or query_lower in c.concept_type.lower()
        ]

        return matches[:limit]

    def get_patterns(
        self, pattern_type: str | None = None, limit: int = 50
    ) -> list[DeveloperPattern]:
        """Get learned patterns.

        Args:
            pattern_type: Optional type filter.
            limit: Maximum results.

        Returns:
            List of patterns.
        """
        return self._db.get_developer_patterns(pattern_type, limit)

    def get_feature_maps(self) -> list[FeatureMap]:
        """Get feature-to-file mappings.

        Returns:
            List of feature maps.
        """
        return self._db.get_feature_maps(self._project_path)

    def get_entry_points(self) -> list[EntryPoint]:
        """Get project entry points.

        Returns:
            List of entry points.
        """
        return self._db.get_entry_points(self._project_path)

    def get_key_directories(self) -> list[KeyDirectory]:
        """Get key project directories.

        Returns:
            List of key directories.
        """
        return self._db.get_key_directories(self._project_path)

    def get_statistics(self) -> dict[str, Any]:
        """Get intelligence statistics.

        Returns:
            Dictionary with statistics about learned intelligence.
        """
        concepts = self._db.get_semantic_concepts()
        patterns = self._db.get_developer_patterns()
        features = self._db.get_feature_maps(self._project_path)

        pattern_stats = self._pattern_engine.get_pattern_statistics()

        return {
            "concepts": {
                "total": len(concepts),
                "by_type": self._count_by_key(concepts, lambda c: c.concept_type),
            },
            "patterns": pattern_stats,
            "features": {
                "total": len(features),
                "names": [f.feature_name for f in features],
            },
        }

    def is_learned(self) -> bool:
        """Check if codebase has been learned.

        Returns:
            True if intelligence exists.
        """
        existing = self._check_existing_intelligence()
        return existing is not None and existing["concepts"] > 0

    def _check_existing_intelligence(self) -> dict[str, int] | None:
        """Check for existing intelligence."""
        concepts = len(self._db.get_semantic_concepts())
        patterns = len(self._db.get_developer_patterns())
        features = len(self._db.get_feature_maps(self._project_path))

        if concepts > 0 or patterns > 0:
            return {"concepts": concepts, "patterns": patterns, "features": features}
        return None

    def _store_project_metadata(self, analysis: CodebaseAnalysisResult) -> None:
        """Store project metadata."""
        existing = self._db.get_project_metadata(self._project_path)
        if not existing:
            self._db.insert_project_metadata(
                ProjectMetadata(
                    project_id=_generate_id(),
                    project_path=self._project_path,
                    project_name=Path(self._project_path).name,
                    language_primary=analysis.languages[0] if analysis.languages else None,
                    languages_detected=analysis.languages,
                    framework_detected=analysis.frameworks,
                    intelligence_version="1.0.0",
                    last_full_scan=datetime.now(),
                )
            )

    def _store_blueprint(self, analysis: CodebaseAnalysisResult) -> None:
        """Store blueprint data (entry points and key directories)."""
        for ep in analysis.entry_points:
            if ep.get("type") and ep.get("filePath"):
                self._db.insert_entry_point(
                    EntryPoint(
                        id=_generate_id(),
                        project_path=self._project_path,
                        entry_type=ep["type"],
                        file_path=ep["filePath"],
                        framework=ep.get("framework"),
                    )
                )

        for kd in analysis.key_directories:
            if kd.get("path") and kd.get("type"):
                self._db.insert_key_directory(
                    KeyDirectory(
                        id=_generate_id(),
                        project_path=self._project_path,
                        directory_path=kd["path"],
                        directory_type=kd["type"],
                        file_count=kd.get("fileCount", 0),
                    )
                )

    def _infer_architecture(self, analysis: CodebaseAnalysisResult) -> str:
        """Infer architecture pattern from analysis."""
        frameworks = [f.lower() for f in analysis.frameworks]
        dirs = {d["type"] for d in analysis.key_directories}

        if any("react" in f for f in frameworks):
            return "Component-Based (React)"
        elif any("svelte" in f for f in frameworks):
            return "Component-Based (Svelte)"
        elif any("express" in f for f in frameworks):
            return "REST API (Express)"
        elif any("fastapi" in f for f in frameworks):
            return "REST API (FastAPI)"
        elif "services" in dirs:
            return "Service-Oriented"
        elif "components" in dirs:
            return "Component-Based"
        elif "models" in dirs and "views" in dirs:
            return "MVC Pattern"
        else:
            return "Modular"

    def _infer_architecture_from_metadata(
        self, metadata: ProjectMetadata, key_dirs: list[KeyDirectory]
    ) -> str:
        """Infer architecture from stored metadata."""
        frameworks = [f.lower() for f in metadata.framework_detected]
        dirs = {kd.directory_type for kd in key_dirs}

        if any("react" in f for f in frameworks):
            return "Component-Based (React)"
        elif any("svelte" in f for f in frameworks):
            return "Component-Based (Svelte)"
        elif any("express" in f for f in frameworks):
            return "REST API (Express)"
        elif any("fastapi" in f for f in frameworks):
            return "REST API (FastAPI)"
        elif "services" in dirs:
            return "Service-Oriented"
        elif "components" in dirs:
            return "Component-Based"
        else:
            return "Modular"

    def _count_by_key(
        self, items: list[Any], key_fn: Callable[[Any], str]
    ) -> dict[str, int]:
        """Count items by key function."""
        counts: dict[str, int] = {}
        for item in items:
            key = key_fn(item)
            counts[key] = counts.get(key, 0) + 1
        return counts


# Convenience function
def learn(
    project_path: str,
    force: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Result[LearningResult, str]:
    """Learn from a codebase.

    Convenience function that creates a client and learns.

    Args:
        project_path: Path to the project.
        force: If True, re-learn even if intelligence exists.
        progress_callback: Optional callback for progress updates.

    Returns:
        Result containing LearningResult on success.
    """
    with InMemoriaClient(project_path) as client:
        return client.learn(force=force, progress_callback=progress_callback)

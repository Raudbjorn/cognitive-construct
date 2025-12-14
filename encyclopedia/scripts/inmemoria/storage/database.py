"""SQLite database for persistent intelligence storage."""

from __future__ import annotations
import json
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..types import (
    SemanticConcept,
    DeveloperPattern,
    FileIntelligence,
    AIInsight,
    FeatureMap,
    EntryPoint,
    KeyDirectory,
    WorkSession,
    ProjectDecision,
    ProjectMetadata,
    LineRange,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Semantic concepts extracted from codebase
CREATE TABLE IF NOT EXISTS semantic_concepts (
  id TEXT PRIMARY KEY,
  concept_name TEXT NOT NULL,
  concept_type TEXT NOT NULL,
  confidence_score REAL DEFAULT 0.0,
  relationships TEXT,
  evolution_history TEXT,
  file_path TEXT,
  line_range TEXT,
  created_at DATETIME DEFAULT (datetime('now', 'utc')),
  updated_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Developer-specific coding patterns
CREATE TABLE IF NOT EXISTS developer_patterns (
  pattern_id TEXT PRIMARY KEY,
  pattern_type TEXT NOT NULL,
  pattern_content TEXT NOT NULL,
  frequency INTEGER DEFAULT 1,
  contexts TEXT,
  examples TEXT,
  confidence REAL DEFAULT 0.0,
  created_at DATETIME DEFAULT (datetime('now', 'utc')),
  last_seen DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Per-file intelligence
CREATE TABLE IF NOT EXISTS file_intelligence (
  file_path TEXT PRIMARY KEY,
  file_hash TEXT NOT NULL,
  semantic_concepts TEXT,
  patterns_used TEXT,
  complexity_metrics TEXT,
  dependencies TEXT,
  last_analyzed DATETIME DEFAULT (datetime('now', 'utc')),
  created_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- AI insights
CREATE TABLE IF NOT EXISTS ai_insights (
  insight_id TEXT PRIMARY KEY,
  insight_type TEXT NOT NULL,
  insight_content TEXT NOT NULL,
  confidence_score REAL DEFAULT 0.0,
  source_agent TEXT,
  validation_status TEXT DEFAULT 'pending',
  impact_prediction TEXT,
  created_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Project metadata
CREATE TABLE IF NOT EXISTS project_metadata (
  project_id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL UNIQUE,
  project_name TEXT,
  language_primary TEXT,
  languages_detected TEXT,
  framework_detected TEXT,
  intelligence_version TEXT,
  last_full_scan DATETIME,
  created_at DATETIME DEFAULT (datetime('now', 'utc')),
  updated_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Feature map
CREATE TABLE IF NOT EXISTS feature_map (
  id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  feature_name TEXT NOT NULL,
  primary_files TEXT NOT NULL,
  related_files TEXT,
  dependencies TEXT,
  status TEXT DEFAULT 'active',
  created_at DATETIME DEFAULT (datetime('now', 'utc')),
  updated_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Entry points
CREATE TABLE IF NOT EXISTS entry_points (
  id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  entry_type TEXT NOT NULL,
  file_path TEXT NOT NULL,
  description TEXT,
  framework TEXT,
  created_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Key directories
CREATE TABLE IF NOT EXISTS key_directories (
  id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  directory_path TEXT NOT NULL,
  directory_type TEXT NOT NULL,
  file_count INTEGER DEFAULT 0,
  description TEXT,
  created_at DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Work sessions
CREATE TABLE IF NOT EXISTS work_sessions (
  id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  session_start DATETIME DEFAULT (datetime('now', 'utc')),
  session_end DATETIME,
  last_feature TEXT,
  current_files TEXT,
  completed_tasks TEXT,
  pending_tasks TEXT,
  blockers TEXT,
  session_notes TEXT,
  last_updated DATETIME DEFAULT (datetime('now', 'utc'))
);

-- Project decisions
CREATE TABLE IF NOT EXISTS project_decisions (
  id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  decision_key TEXT NOT NULL,
  decision_value TEXT NOT NULL,
  reasoning TEXT,
  made_at DATETIME DEFAULT (datetime('now', 'utc')),
  UNIQUE(project_path, decision_key)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_type ON semantic_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_file ON semantic_concepts(file_path);
CREATE INDEX IF NOT EXISTS idx_developer_patterns_type ON developer_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_developer_patterns_frequency ON developer_patterns(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_file_intelligence_analyzed ON file_intelligence(last_analyzed);
CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_feature_map_project ON feature_map(project_path);
CREATE INDEX IF NOT EXISTS idx_feature_map_name ON feature_map(feature_name);
CREATE INDEX IF NOT EXISTS idx_entry_points_project ON entry_points(project_path);
CREATE INDEX IF NOT EXISTS idx_key_directories_project ON key_directories(project_path);
CREATE INDEX IF NOT EXISTS idx_work_sessions_project ON work_sessions(project_path);
CREATE INDEX IF NOT EXISTS idx_project_decisions_key ON project_decisions(project_path, decision_key);
"""


class Database:
    """SQLite database for persistent intelligence storage."""

    def __init__(self, db_path: str | Path = ":memory:"):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory.
        """
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Get database cursor with automatic commit/rollback."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime from SQLite."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace(" ", "T"))
        except ValueError:
            return None

    # Semantic Concepts
    def insert_semantic_concept(self, concept: SemanticConcept) -> None:
        """Insert or update a semantic concept."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO semantic_concepts (
                    id, concept_name, concept_type, confidence_score,
                    relationships, evolution_history, file_path, line_range
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept.id,
                    concept.concept_name,
                    concept.concept_type,
                    concept.confidence_score,
                    json.dumps(concept.relationships),
                    json.dumps(concept.evolution_history),
                    concept.file_path,
                    json.dumps({"start": concept.line_range.start, "end": concept.line_range.end}),
                ),
            )

    def get_semantic_concepts(self, file_path: str | None = None) -> list[SemanticConcept]:
        """Get semantic concepts, optionally filtered by file path."""
        with self._cursor() as cur:
            if file_path:
                cur.execute("SELECT * FROM semantic_concepts WHERE file_path = ?", (file_path,))
            else:
                cur.execute("SELECT * FROM semantic_concepts")

            return [self._row_to_concept(row) for row in cur.fetchall()]

    def _row_to_concept(self, row: sqlite3.Row) -> SemanticConcept:
        """Convert database row to SemanticConcept."""
        line_range = json.loads(row["line_range"] or '{"start": 0, "end": 0}')
        return SemanticConcept(
            id=row["id"],
            concept_name=row["concept_name"],
            concept_type=row["concept_type"],
            confidence_score=row["confidence_score"],
            relationships=json.loads(row["relationships"] or "{}"),
            evolution_history=json.loads(row["evolution_history"] or "{}"),
            file_path=row["file_path"],
            line_range=LineRange(start=line_range["start"], end=line_range["end"]),
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
        )

    # Developer Patterns
    def insert_developer_pattern(self, pattern: DeveloperPattern) -> None:
        """Insert or update a developer pattern."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO developer_patterns (
                    pattern_id, pattern_type, pattern_content, frequency,
                    contexts, examples, confidence, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'utc'))
                """,
                (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.pattern_content),
                    pattern.frequency,
                    json.dumps(pattern.contexts),
                    json.dumps(pattern.examples),
                    pattern.confidence,
                ),
            )

    def get_developer_patterns(
        self, pattern_type: str | None = None, limit: int = 50
    ) -> list[DeveloperPattern]:
        """Get developer patterns, optionally filtered by type."""
        with self._cursor() as cur:
            query = "SELECT * FROM developer_patterns"
            params: list[Any] = []

            if pattern_type:
                query += " WHERE pattern_type = ?"
                params.append(pattern_type)

            query += " ORDER BY frequency DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            return [self._row_to_pattern(row) for row in cur.fetchall()]

    def _row_to_pattern(self, row: sqlite3.Row) -> DeveloperPattern:
        """Convert database row to DeveloperPattern."""
        return DeveloperPattern(
            pattern_id=row["pattern_id"],
            pattern_type=row["pattern_type"],
            pattern_content=json.loads(row["pattern_content"]),
            frequency=row["frequency"],
            contexts=json.loads(row["contexts"] or "[]"),
            examples=json.loads(row["examples"] or "[]"),
            confidence=row["confidence"],
            created_at=self._parse_datetime(row["created_at"]),
            last_seen=self._parse_datetime(row["last_seen"]),
        )

    # Feature Maps
    def insert_feature_map(self, feature: FeatureMap) -> None:
        """Insert or update a feature map."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO feature_map (
                    id, project_path, feature_name, primary_files,
                    related_files, dependencies, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature.id,
                    feature.project_path,
                    feature.feature_name,
                    json.dumps(feature.primary_files),
                    json.dumps(feature.related_files),
                    json.dumps(feature.dependencies),
                    feature.status,
                ),
            )

    def get_feature_maps(self, project_path: str) -> list[FeatureMap]:
        """Get feature maps for a project."""
        with self._cursor() as cur:
            # Try both absolute and relative paths
            cur.execute(
                """
                SELECT * FROM feature_map
                WHERE (project_path = ? OR project_path = '.') AND status = 'active'
                ORDER BY feature_name
                """,
                (project_path,),
            )
            return [self._row_to_feature_map(row) for row in cur.fetchall()]

    def search_feature_maps(self, project_path: str, query: str) -> list[FeatureMap]:
        """Search feature maps by name."""
        with self._cursor() as cur:
            pattern = f"%{query}%"
            cur.execute(
                """
                SELECT * FROM feature_map
                WHERE project_path = ? AND status = 'active'
                AND feature_name LIKE ?
                ORDER BY feature_name
                """,
                (project_path, pattern),
            )
            return [self._row_to_feature_map(row) for row in cur.fetchall()]

    def _row_to_feature_map(self, row: sqlite3.Row) -> FeatureMap:
        """Convert database row to FeatureMap."""
        return FeatureMap(
            id=row["id"],
            project_path=row["project_path"],
            feature_name=row["feature_name"],
            primary_files=json.loads(row["primary_files"] or "[]"),
            related_files=json.loads(row["related_files"] or "[]"),
            dependencies=json.loads(row["dependencies"] or "[]"),
            status=row["status"],
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
        )

    # Entry Points
    def insert_entry_point(self, entry_point: EntryPoint) -> None:
        """Insert or update an entry point."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO entry_points (
                    id, project_path, entry_type, file_path, description, framework
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_point.id,
                    entry_point.project_path,
                    entry_point.entry_type,
                    entry_point.file_path,
                    entry_point.description,
                    entry_point.framework,
                ),
            )

    def get_entry_points(self, project_path: str) -> list[EntryPoint]:
        """Get entry points for a project."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM entry_points
                WHERE project_path = ? OR project_path = '.'
                ORDER BY entry_type, file_path
                """,
                (project_path,),
            )
            return [
                EntryPoint(
                    id=row["id"],
                    project_path=row["project_path"],
                    entry_type=row["entry_type"],
                    file_path=row["file_path"],
                    description=row["description"],
                    framework=row["framework"],
                    created_at=self._parse_datetime(row["created_at"]),
                )
                for row in cur.fetchall()
            ]

    # Key Directories
    def insert_key_directory(self, directory: KeyDirectory) -> None:
        """Insert or update a key directory."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO key_directories (
                    id, project_path, directory_path, directory_type,
                    file_count, description
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    directory.id,
                    directory.project_path,
                    directory.directory_path,
                    directory.directory_type,
                    directory.file_count,
                    directory.description,
                ),
            )

    def get_key_directories(self, project_path: str) -> list[KeyDirectory]:
        """Get key directories for a project."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM key_directories
                WHERE project_path = ? OR project_path = '.'
                ORDER BY directory_type, directory_path
                """,
                (project_path,),
            )
            return [
                KeyDirectory(
                    id=row["id"],
                    project_path=row["project_path"],
                    directory_path=row["directory_path"],
                    directory_type=row["directory_type"],
                    file_count=row["file_count"],
                    description=row["description"],
                    created_at=self._parse_datetime(row["created_at"]),
                )
                for row in cur.fetchall()
            ]

    # Project Metadata
    def insert_project_metadata(self, metadata: ProjectMetadata) -> None:
        """Insert or update project metadata."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO project_metadata (
                    project_id, project_path, project_name, language_primary,
                    languages_detected, framework_detected, intelligence_version,
                    last_full_scan
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.project_id,
                    metadata.project_path,
                    metadata.project_name,
                    metadata.language_primary,
                    json.dumps(metadata.languages_detected),
                    json.dumps(metadata.framework_detected),
                    metadata.intelligence_version,
                    metadata.last_full_scan.isoformat() if metadata.last_full_scan else None,
                ),
            )

    def get_project_metadata(self, project_path: str) -> ProjectMetadata | None:
        """Get project metadata."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM project_metadata WHERE project_path = ? LIMIT 1",
                (project_path,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return ProjectMetadata(
                project_id=row["project_id"],
                project_path=row["project_path"],
                project_name=row["project_name"],
                language_primary=row["language_primary"],
                languages_detected=json.loads(row["languages_detected"] or "[]"),
                framework_detected=json.loads(row["framework_detected"] or "[]"),
                intelligence_version=row["intelligence_version"],
                last_full_scan=self._parse_datetime(row["last_full_scan"]),
                created_at=self._parse_datetime(row["created_at"]),
                updated_at=self._parse_datetime(row["updated_at"]),
            )

    # Work Sessions
    def create_work_session(self, session: WorkSession) -> None:
        """Create a new work session."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO work_sessions (
                    id, project_path, session_end, last_feature, current_files,
                    completed_tasks, pending_tasks, blockers, session_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.project_path,
                    session.session_end.isoformat() if session.session_end else None,
                    session.last_feature,
                    json.dumps(session.current_files),
                    json.dumps(session.completed_tasks),
                    json.dumps(session.pending_tasks),
                    json.dumps(session.blockers),
                    session.session_notes,
                ),
            )

    def get_current_work_session(self, project_path: str) -> WorkSession | None:
        """Get current (open) work session."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM work_sessions
                WHERE project_path = ? AND session_end IS NULL
                ORDER BY session_start DESC LIMIT 1
                """,
                (project_path,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return WorkSession(
                id=row["id"],
                project_path=row["project_path"],
                session_start=self._parse_datetime(row["session_start"]) or datetime.now(),
                session_end=self._parse_datetime(row["session_end"]),
                last_feature=row["last_feature"],
                current_files=json.loads(row["current_files"] or "[]"),
                completed_tasks=json.loads(row["completed_tasks"] or "[]"),
                pending_tasks=json.loads(row["pending_tasks"] or "[]"),
                blockers=json.loads(row["blockers"] or "[]"),
                session_notes=row["session_notes"],
                last_updated=self._parse_datetime(row["last_updated"]),
            )

    # Project Decisions
    def upsert_project_decision(self, decision: ProjectDecision) -> None:
        """Insert or update a project decision."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO project_decisions (
                    id, project_path, decision_key, decision_value, reasoning
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.project_path,
                    decision.decision_key,
                    decision.decision_value,
                    decision.reasoning,
                ),
            )

    def get_project_decisions(self, project_path: str, limit: int = 20) -> list[ProjectDecision]:
        """Get project decisions."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM project_decisions
                WHERE project_path = ?
                ORDER BY made_at DESC LIMIT ?
                """,
                (project_path, limit),
            )
            return [
                ProjectDecision(
                    id=row["id"],
                    project_path=row["project_path"],
                    decision_key=row["decision_key"],
                    decision_value=row["decision_value"],
                    reasoning=row["reasoning"],
                    made_at=self._parse_datetime(row["made_at"]),
                )
                for row in cur.fetchall()
            ]

    def get_project_decision(self, project_path: str, decision_key: str) -> ProjectDecision | None:
        """Get a specific project decision."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM project_decisions
                WHERE project_path = ? AND decision_key = ?
                """,
                (project_path, decision_key),
            )
            row = cur.fetchone()
            if not row:
                return None

            return ProjectDecision(
                id=row["id"],
                project_path=row["project_path"],
                decision_key=row["decision_key"],
                decision_value=row["decision_value"],
                reasoning=row["reasoning"],
                made_at=self._parse_datetime(row["made_at"]),
            )

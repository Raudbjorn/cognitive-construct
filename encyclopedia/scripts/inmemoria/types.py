"""Type definitions for inmemoria codebase intelligence library."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ConceptType(str, Enum):
    """Types of semantic concepts."""
    CLASS = "class"
    FUNCTION = "function"
    INTERFACE = "interface"
    MODULE = "module"
    PATTERN = "pattern"
    VARIABLE = "variable"
    IMPORT = "import"
    EXPORT = "export"


class PatternType(str, Enum):
    """Types of coding patterns."""
    NAMING = "naming"
    STRUCTURE = "structure"
    IMPLEMENTATION = "implementation"
    STYLE = "style"
    TESTING = "testing"
    API_DESIGN = "api_design"
    DEPENDENCY_INJECTION = "dependency_injection"
    FACTORY = "factory"
    SINGLETON = "singleton"
    OBSERVER = "observer"


class ValidationStatus(str, Enum):
    """Validation status for AI insights."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


class WorkType(str, Enum):
    """Type of work being performed."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"


class Complexity(str, Enum):
    """Complexity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Core data types
@dataclass(slots=True)
class LineRange:
    """Line range in a file."""
    start: int
    end: int


@dataclass(slots=True)
class SemanticConcept:
    """A semantic concept extracted from code."""
    id: str
    concept_name: str
    concept_type: str
    confidence_score: float
    relationships: dict[str, Any]
    evolution_history: dict[str, Any]
    file_path: str
    line_range: LineRange
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class DeveloperPattern:
    """A coding pattern learned from the codebase."""
    pattern_id: str
    pattern_type: str
    pattern_content: dict[str, Any]
    frequency: int
    contexts: list[str]
    examples: list[dict[str, Any]]
    confidence: float
    created_at: datetime | None = None
    last_seen: datetime | None = None


@dataclass(slots=True)
class FileIntelligence:
    """Per-file intelligence and analysis results."""
    file_path: str
    file_hash: str
    semantic_concepts: list[str]
    patterns_used: list[str]
    complexity_metrics: dict[str, float]
    dependencies: list[str]
    last_analyzed: datetime | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class AIInsight:
    """AI-generated insight about the codebase."""
    insight_id: str
    insight_type: str
    insight_content: dict[str, Any]
    confidence_score: float
    source_agent: str
    validation_status: ValidationStatus
    impact_prediction: dict[str, Any]
    created_at: datetime | None = None


@dataclass(slots=True)
class FeatureMap:
    """Mapping of features to files."""
    id: str
    project_path: str
    feature_name: str
    primary_files: list[str]
    related_files: list[str]
    dependencies: list[str]
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class EntryPoint:
    """Project entry point."""
    id: str
    project_path: str
    entry_type: str  # 'web', 'api', 'cli', 'worker'
    file_path: str
    description: str | None = None
    framework: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class KeyDirectory:
    """Important project directory."""
    id: str
    project_path: str
    directory_path: str
    directory_type: str  # 'components', 'utils', 'services', etc.
    file_count: int = 0
    description: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class WorkSession:
    """Work session tracking."""
    id: str
    project_path: str
    session_start: datetime
    session_end: datetime | None = None
    last_feature: str | None = None
    current_files: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    session_notes: str | None = None
    last_updated: datetime | None = None


@dataclass(slots=True)
class ProjectDecision:
    """Architectural decision record."""
    id: str
    project_path: str
    decision_key: str
    decision_value: str
    reasoning: str | None = None
    made_at: datetime | None = None


@dataclass(slots=True)
class ProjectMetadata:
    """Project metadata."""
    project_id: str
    project_path: str
    project_name: str | None = None
    language_primary: str | None = None
    languages_detected: list[str] = field(default_factory=list)
    framework_detected: list[str] = field(default_factory=list)
    intelligence_version: str | None = None
    last_full_scan: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# Analysis result types
@dataclass(slots=True)
class ComplexityMetrics:
    """Complexity metrics for code."""
    cyclomatic: float
    cognitive: float
    lines: int


@dataclass(slots=True)
class CodebaseAnalysisResult:
    """Result of codebase analysis."""
    languages: list[str]
    frameworks: list[str]
    complexity: ComplexityMetrics
    concepts: list[dict[str, Any]]
    analysis_status: str = "normal"  # 'normal' or 'degraded'
    errors: list[str] = field(default_factory=list)
    entry_points: list[dict[str, Any]] = field(default_factory=list)
    key_directories: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PatternAnalysisResult:
    """Result of pattern analysis."""
    detected: list[str]
    violations: list[str]
    recommendations: list[str]
    learned: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RelevantPattern:
    """A pattern relevant to the current task."""
    pattern_id: str
    pattern_type: str
    pattern_content: dict[str, Any]
    frequency: int
    contexts: list[str]
    examples: list[dict[str, str]]
    confidence: float


@dataclass(slots=True)
class ApproachPrediction:
    """Predicted approach for a problem."""
    approach: str
    confidence: float
    reasoning: str
    patterns: list[str]
    complexity: Complexity


@dataclass(slots=True)
class FileRouting:
    """Result of routing a request to files."""
    intended_feature: str
    target_files: list[str]
    work_type: WorkType
    suggested_start_point: str
    confidence: float
    reasoning: str


@dataclass(slots=True)
class Blueprint:
    """Project blueprint summary."""
    tech_stack: list[str]
    entry_points: dict[str, str]
    key_directories: dict[str, str]
    architecture: str


@dataclass(slots=True)
class LearningResult:
    """Result of learning from a codebase."""
    success: bool
    concepts_learned: int
    patterns_learned: int
    features_learned: int
    insights: list[str]
    time_elapsed_ms: int
    blueprint: Blueprint | None = None

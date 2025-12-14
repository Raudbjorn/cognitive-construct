"""
inmemoria - Python library for persistent codebase intelligence

A library that learns from your codebase and provides intelligent
code understanding, pattern detection, and file routing capabilities.

Usage:
    from inmemoria import InMemoriaClient

    # Initialize client for a project
    client = InMemoriaClient("/path/to/project")

    # Learn from codebase (one-time, ~30-60 seconds)
    result = client.learn()
    if result.is_ok():
        print(f"Learned {result.value.concepts_learned} concepts")
        print(f"Learned {result.value.patterns_learned} patterns")

    # Get project blueprint
    blueprint = client.get_blueprint()
    print(f"Tech stack: {blueprint.tech_stack}")
    print(f"Entry points: {blueprint.entry_points}")

    # Find relevant patterns for a task
    patterns = client.find_patterns("add authentication")
    for p in patterns:
        print(f"{p.pattern_type}: {p.pattern_content}")

    # Predict approach for a problem
    approach = client.predict_approach("implement user registration")
    print(f"Suggested approach: {approach.approach}")
    print(f"Confidence: {approach.confidence}")

    # Route a task to files
    routing = client.route_to_files("fix login bug")
    if routing:
        print(f"Feature: {routing.intended_feature}")
        print(f"Start at: {routing.suggested_start_point}")
        print(f"Files: {routing.target_files}")

    # Close when done
    client.close()

    # Or use context manager
    with InMemoriaClient("/path/to/project") as client:
        if not client.is_learned():
            client.learn()
        blueprint = client.get_blueprint()
"""

from .client import InMemoriaClient, learn
from .config import Config, get_config, set_config
from .result import Result, Ok, Err
from .types import (
    # Core types
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
    # Enums
    ConceptType,
    PatternType,
    ValidationStatus,
    WorkType,
    Complexity,
    # Result types
    ComplexityMetrics,
    CodebaseAnalysisResult,
    PatternAnalysisResult,
    RelevantPattern,
    ApproachPrediction,
    FileRouting,
    Blueprint,
    LearningResult,
)

__all__ = [
    # Main client
    "InMemoriaClient",
    "learn",
    # Config
    "Config",
    "get_config",
    "set_config",
    # Result types
    "Result",
    "Ok",
    "Err",
    # Core types
    "SemanticConcept",
    "DeveloperPattern",
    "FileIntelligence",
    "AIInsight",
    "FeatureMap",
    "EntryPoint",
    "KeyDirectory",
    "WorkSession",
    "ProjectDecision",
    "ProjectMetadata",
    "LineRange",
    # Enums
    "ConceptType",
    "PatternType",
    "ValidationStatus",
    "WorkType",
    "Complexity",
    # Result types
    "ComplexityMetrics",
    "CodebaseAnalysisResult",
    "PatternAnalysisResult",
    "RelevantPattern",
    "ApproachPrediction",
    "FileRouting",
    "Blueprint",
    "LearningResult",
]

__version__ = "1.0.0"

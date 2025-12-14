"""Type definitions for the Semgrep MCP client."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# === Enums ===


class TriageState(str, Enum):
    """Triage state for findings."""

    UNTRIAGED = "untriaged"
    IGNORED = "ignored"
    TRIAGED = "triaged"


class Severity(str, Enum):
    """Severity levels for findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Confidence(str, Enum):
    """Confidence levels for findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# === Request Types ===


@dataclass(frozen=True, slots=True)
class CodeFile:
    """Code file for scanning.

    Attributes:
        path: Path of the code file (may be virtual for remote scanning)
        content: Content of the code file
    """

    path: str
    content: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeFile:
        """Create CodeFile from dictionary."""
        return cls(
            path=data.get("path", ""),
            content=data.get("content", ""),
        )


@dataclass(frozen=True, slots=True)
class CodeWithLanguage:
    """Code content with explicit language specification."""

    content: str
    language: str = "python"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeWithLanguage:
        """Create CodeWithLanguage from dictionary."""
        return cls(
            content=data.get("content", ""),
            language=data.get("language", "python"),
        )


# === Response Types ===


@dataclass(frozen=True, slots=True)
class SemgrepScanResult:
    """Result from a Semgrep scan operation."""

    version: str
    results: list[dict[str, Any]]
    paths: dict[str, Any]
    errors: list[dict[str, Any]] = field(default_factory=list)
    skipped_rules: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemgrepScanResult:
        """Create SemgrepScanResult from dictionary."""
        return cls(
            version=data.get("version", ""),
            results=data.get("results", []),
            paths=data.get("paths", {}),
            errors=data.get("errors", []),
            skipped_rules=data.get("skipped_rules", []),
        )


@dataclass(frozen=True, slots=True)
class Location:
    """Source location for a finding."""

    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Location:
        """Create Location from dictionary."""
        return cls(
            file_path=data.get("file_path", ""),
            line=data.get("line", 0),
            column=data.get("column", 0),
            end_line=data.get("end_line", 0),
            end_column=data.get("end_column", 0),
        )


@dataclass(frozen=True, slots=True)
class Repository:
    """Repository information."""

    name: str
    url: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Repository:
        """Create Repository from dictionary."""
        return cls(
            name=data.get("name", ""),
            url=str(data.get("url", "")),
        )


@dataclass(frozen=True, slots=True)
class ExternalTicket:
    """External ticket reference."""

    external_slug: str
    url: str
    id: int
    linked_issue_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalTicket:
        """Create ExternalTicket from dictionary."""
        return cls(
            external_slug=data.get("external_slug", ""),
            url=str(data.get("url", "")),
            id=data.get("id", 0),
            linked_issue_ids=data.get("linked_issue_ids", []),
        )


@dataclass(frozen=True, slots=True)
class ReviewComment:
    """Review comment reference."""

    external_discussion_id: str
    external_note_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewComment:
        """Create ReviewComment from dictionary."""
        return cls(
            external_discussion_id=data.get("external_discussion_id", ""),
            external_note_id=data.get("external_note_id"),
        )


@dataclass(frozen=True, slots=True)
class SourcingPolicy:
    """Sourcing policy information."""

    id: int
    name: str
    slug: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourcingPolicy:
        """Create SourcingPolicy from dictionary."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
        )


@dataclass(frozen=True, slots=True)
class Rule:
    """Semgrep rule information."""

    name: str
    message: str
    confidence: str
    category: str
    subcategories: list[str] = field(default_factory=list)
    vulnerability_classes: list[str] = field(default_factory=list)
    cwe_names: list[str] = field(default_factory=list)
    owasp_names: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        """Create Rule from dictionary."""
        return cls(
            name=data.get("name", ""),
            message=data.get("message", ""),
            confidence=data.get("confidence", ""),
            category=data.get("category", ""),
            subcategories=data.get("subcategories", []),
            vulnerability_classes=data.get("vulnerability_classes", []),
            cwe_names=data.get("cwe_names", []),
            owasp_names=data.get("owasp_names", []),
        )


@dataclass(frozen=True, slots=True)
class Autofix:
    """Autofix suggestion from Semgrep Assistant."""

    fix_code: str
    explanation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Autofix:
        """Create Autofix from dictionary."""
        return cls(
            fix_code=data.get("fix_code", ""),
            explanation=data.get("explanation", ""),
        )


@dataclass(frozen=True, slots=True)
class Guidance:
    """Guidance from Semgrep Assistant."""

    summary: str
    instructions: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Guidance:
        """Create Guidance from dictionary."""
        return cls(
            summary=data.get("summary", ""),
            instructions=data.get("instructions", ""),
        )


@dataclass(frozen=True, slots=True)
class Autotriage:
    """Autotriage result from Semgrep Assistant."""

    verdict: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Autotriage:
        """Create Autotriage from dictionary."""
        return cls(
            verdict=data.get("verdict", ""),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True, slots=True)
class Component:
    """Component classification from Semgrep Assistant."""

    tag: str
    risk: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Component:
        """Create Component from dictionary."""
        return cls(
            tag=data.get("tag", ""),
            risk=data.get("risk", ""),
        )


@dataclass(frozen=True, slots=True)
class Assistant:
    """Semgrep Assistant analysis results."""

    autofix: Autofix | None = None
    guidance: Guidance | None = None
    autotriage: Autotriage | None = None
    component: Component | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assistant:
        """Create Assistant from dictionary."""
        return cls(
            autofix=Autofix.from_dict(data["autofix"]) if data.get("autofix") else None,
            guidance=Guidance.from_dict(data["guidance"]) if data.get("guidance") else None,
            autotriage=Autotriage.from_dict(data["autotriage"]) if data.get("autotriage") else None,
            component=Component.from_dict(data["component"]) if data.get("component") else None,
        )


@dataclass(frozen=True, slots=True)
class Finding:
    """Semgrep finding from a scan or API."""

    id: int
    ref: str
    first_seen_scan_id: int
    syntactic_id: str
    match_based_id: str
    repository: Repository
    line_of_code_url: str
    triage_state: str
    state: str
    status: str
    severity: str
    confidence: str
    categories: list[str]
    created_at: datetime
    relevant_since: datetime
    rule_name: str
    rule_message: str
    location: Location
    state_updated_at: datetime
    rule: Rule
    review_comments: list[ReviewComment] = field(default_factory=list)
    external_ticket: ExternalTicket | None = None
    sourcing_policy: SourcingPolicy | None = None
    triaged_at: datetime | None = None
    triage_comment: str | None = None
    triage_reason: str | None = None
    assistant: Assistant | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Create Finding from dictionary."""
        return cls(
            id=data.get("id", 0),
            ref=data.get("ref", ""),
            first_seen_scan_id=data.get("first_seen_scan_id", 0),
            syntactic_id=data.get("syntactic_id", ""),
            match_based_id=data.get("match_based_id", ""),
            repository=Repository.from_dict(data.get("repository", {})),
            line_of_code_url=str(data.get("line_of_code_url", "")),
            triage_state=data.get("triage_state", ""),
            state=data.get("state", ""),
            status=data.get("status", ""),
            severity=data.get("severity", ""),
            confidence=data.get("confidence", ""),
            categories=data.get("categories", []),
            created_at=_parse_datetime(data.get("created_at")),
            relevant_since=_parse_datetime(data.get("relevant_since")),
            rule_name=data.get("rule_name", ""),
            rule_message=data.get("rule_message", ""),
            location=Location.from_dict(data.get("location", {})),
            state_updated_at=_parse_datetime(data.get("state_updated_at")),
            rule=Rule.from_dict(data.get("rule", {})),
            review_comments=[ReviewComment.from_dict(rc) for rc in data.get("review_comments", [])],
            external_ticket=(
                ExternalTicket.from_dict(data["external_ticket"])
                if data.get("external_ticket")
                else None
            ),
            sourcing_policy=(
                SourcingPolicy.from_dict(data["sourcing_policy"])
                if data.get("sourcing_policy")
                else None
            ),
            triaged_at=_parse_datetime(data.get("triaged_at")) if data.get("triaged_at") else None,
            triage_comment=data.get("triage_comment"),
            triage_reason=data.get("triage_reason"),
            assistant=Assistant.from_dict(data["assistant"]) if data.get("assistant") else None,
        )


# === Error Types ===


@dataclass(frozen=True, slots=True)
class SemgrepError:
    """Semgrep API/CLI error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


# === Helper Functions ===


def _parse_datetime(value: str | datetime | None) -> datetime:
    """Parse datetime from string or return as-is."""
    if value is None:
        return datetime.min
    if isinstance(value, datetime):
        return value
    try:
        # ISO format parsing
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.min

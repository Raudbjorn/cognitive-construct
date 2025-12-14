"""Type definitions for the Shodan client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# === Response Types ===

@dataclass(frozen=True, slots=True)
class Location:
    """Geographic location information."""

    country_name: str
    city: str | None = None


@dataclass(frozen=True, slots=True)
class ShodanSearchMatch:
    """A single match from a Shodan search."""

    ip_str: str
    port: int
    org: str
    location: Location
    hostnames: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    product: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShodanSearchResult:
    """Search results from Shodan."""

    total: int
    matches: list[ShodanSearchMatch]


@dataclass(frozen=True, slots=True)
class ShodanHostResult:
    """Host information from Shodan."""

    ip_str: str
    org: str
    isp: str
    asn: str
    country_name: str
    city: str | None
    ports: list[int] = field(default_factory=list)
    hostnames: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CveInfo:
    """CVE vulnerability information."""

    cve_id: str
    summary: str | None = None
    cvss_v3: float | None = None
    cvss_v2: float | None = None
    epss: float | None = None
    kev: bool = False
    propose_action: str | None = None
    ransomware_campaign: str | None = None
    published_time: str | None = None
    references: list[str] = field(default_factory=list)


# === Error Types ===

@dataclass(frozen=True, slots=True)
class ApiError:
    """API error details."""

    message: str
    code: str | None = None
    status_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

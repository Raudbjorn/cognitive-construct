"""Shodan API client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import (
    API_BASE_URL,
    CVEDB_API_URL,
    DEFAULT_TIMEOUT,
    API_KEY_ENV_VAR,
)
from .result import Err, Ok, Result
from .types import (
    ApiError,
    CveInfo,
    Location,
    ShodanHostResult,
    ShodanSearchMatch,
    ShodanSearchResult,
)


@dataclass
class ShodanClient:
    """Client for Shodan API."""

    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _resolved_key: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize resolved API key."""
        self._resolved_key = self.api_key or os.environ.get(API_KEY_ENV_VAR)

    def _check_api_key(self) -> Result[None, ApiError]:
        """Verify API key is configured."""
        if not self._resolved_key:
            return Err(
                ApiError(
                    message=f"API key required. Set {API_KEY_ENV_VAR} or pass api_key parameter.",
                    code="AUTH_REQUIRED",
                )
            )
        return Ok(None)

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Result[dict[str, Any], ApiError]:
        """Make HTTP request with error handling."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                )

                if not response.is_success:
                    return Err(
                        ApiError(
                            message=f"Shodan API error: {response.text}",
                            status_code=response.status_code,
                        )
                    )

                return Ok(response.json())

        except httpx.TimeoutException:
            return Err(ApiError(message="Request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(ApiError(message=f"Request failed: {e}", code="REQUEST_ERROR"))

    async def search(self, query: str, limit: int = 10) -> Result[ShodanSearchResult, ApiError]:
        """Search Shodan."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{API_BASE_URL}/shodan/host/search"
        result = await self._request(
            "GET",
            url,
            params={"key": self._resolved_key, "query": query, "limit": limit},
        )

        if result.is_err():
            return result  # type: ignore

        data = result.value
        matches = []
        for m in data.get("matches", []):
            loc = m.get("location", {})
            matches.append(ShodanSearchMatch(
                ip_str=m.get("ip_str", ""),
                port=m.get("port", 0),
                org=m.get("org", ""),
                location=Location(
                    country_name=loc.get("country_name", ""),
                    city=loc.get("city")
                ),
                hostnames=m.get("hostnames", []),
                domains=m.get("domains", []),
                product=m.get("product"),
                metadata={k: v for k, v in m.items() if k not in ["ip_str", "port", "org", "location", "hostnames", "domains", "product"]}
            ))

        return Ok(ShodanSearchResult(
            total=data.get("total", 0),
            matches=matches
        ))

    async def host(self, ip: str) -> Result[ShodanHostResult, ApiError]:
        """Get host info."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{API_BASE_URL}/shodan/host/{ip}"
        result = await self._request(
            "GET",
            url,
            params={"key": self._resolved_key},
        )

        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(ShodanHostResult(
            ip_str=data.get("ip_str", ""),
            org=data.get("org", ""),
            isp=data.get("isp", ""),
            asn=data.get("asn", ""),
            country_name=data.get("country_name", ""),
            city=data.get("city"),
            ports=data.get("ports", []),
            hostnames=data.get("hostnames", []),
            domains=data.get("domains", []),
            tags=data.get("tags", []),
            metadata={k: v for k, v in data.items() if k not in ["ip_str", "org", "isp", "asn", "country_name", "city", "ports", "hostnames", "domains", "tags"]}
        ))

    async def dns_resolve(self, hostnames: list[str]) -> Result[dict[str, str], ApiError]:
        """Resolve hostnames."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{API_BASE_URL}/dns/resolve"
        result = await self._request(
            "GET",
            url,
            params={"key": self._resolved_key, "hostnames": ",".join(hostnames)},
        )
        
        if result.is_err():
            return result  # type: ignore
            
        return Ok(result.value)

    async def dns_reverse(self, ips: list[str]) -> Result[dict[str, list[str]], ApiError]:
        """Reverse DNS lookup."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{API_BASE_URL}/dns/reverse"
        result = await self._request(
            "GET",
            url,
            params={"key": self._resolved_key, "ips": ",".join(ips)},
        )

        if result.is_err():
            return result  # type: ignore

        return Ok(result.value)

    async def cve(self, cve_id: str) -> Result[CveInfo, ApiError]:
        """Get CVE info."""
        # No API Key needed for CVEDB? The original code didn't use one for CVEDB.
        # But `_get_httpx` implies it uses the same client setup. 
        # `cve_lookup` in original used `httpx.get(url)` without params.
        
        url = f"{CVEDB_API_URL}/cve/{cve_id}"
        result = await self._request("GET", url)

        if result.is_err():
            # Handle 404 specifically if needed, but _request returns generic error
            return result  # type: ignore

        data = result.value
        return Ok(CveInfo(
            cve_id=data.get("cve_id", cve_id),
            summary=data.get("summary"),
            cvss_v3=data.get("cvss_v3"),
            cvss_v2=data.get("cvss_v2"),
            epss=data.get("epss"),
            kev=data.get("kev", False),
            propose_action=data.get("propose_action"),
            ransomware_campaign=data.get("ransomware_campaign"),
            published_time=data.get("published_time"),
            references=data.get("references", [])
        ))

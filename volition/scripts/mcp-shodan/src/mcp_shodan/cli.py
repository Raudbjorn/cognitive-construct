"""Shodan CLI implementation."""

import argparse
import asyncio
import json
import sys

from .client import ShodanClient


def redact_ip(ip: str) -> str:
    """Partially redact an IP address for security."""
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.xxx.xxx"
    return ip


def format_cve_severity(score: float) -> str:
    """Format CVSS score as severity level."""
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    if score >= 0.1:
        return "Low"
    return "None"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shodan - Security reconnaissance API wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m mcp_shodan search "apache port:443" --confirm
    python -m mcp_shodan host 8.8.8.8
    python -m mcp_shodan dns google.com
    python -m mcp_shodan reverse 8.8.8.8
    python -m mcp_shodan cve CVE-2021-44228

Security Notes:
    - Search queries require --confirm flag
    - IP addresses are partially redacted by default
    - All queries are subject to Shodan's ToS
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search command
    search_parser = subparsers.add_parser("search", help="Search Shodan database")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-y", "--confirm",
        action="store_true",
        help="Confirm the query (required for search)",
    )
    search_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Limit results (default: 10)",
    )
    search_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    search_parser.add_argument("--no-redact", action="store_true", help="Don't redact IP addresses")

    # host command
    host_parser = subparsers.add_parser("host", help="Get info about an IP address")
    host_parser.add_argument("ip", help="IP address to look up")
    host_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    host_parser.add_argument("--no-redact", action="store_true", help="Don't redact IP addresses")

    # dns command
    dns_parser = subparsers.add_parser("dns", help="Resolve hostname to IP")
    dns_parser.add_argument("hostname", help="Hostname(s) to resolve (comma-separated)")
    dns_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    # reverse command
    reverse_parser = subparsers.add_parser("reverse", help="Reverse DNS lookup")
    reverse_parser.add_argument("ip", help="IP address(es) for reverse lookup (comma-separated)")
    reverse_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    # cve command
    cve_parser = subparsers.add_parser("cve", help="Look up CVE information")
    cve_parser.add_argument("cve_id", help="CVE ID (e.g., CVE-2021-44228)")
    cve_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    client = ShodanClient()

    if args.command == "search":
        if not args.confirm:
            print("Security Notice: Shodan searches may reveal sensitive infrastructure information.", file=sys.stderr)
            print("This action is logged and subject to audit.", file=sys.stderr)
            print(file=sys.stderr)
            print(f"Query: {args.query}", file=sys.stderr)
            print(file=sys.stderr)
            print("Re-run with --confirm flag to proceed.", file=sys.stderr)
            sys.exit(1)

        result = asyncio.run(client.search(args.query, args.limit))

        if result.is_err():
            print(f"Error: {result.error.message}", file=sys.stderr)
            sys.exit(1)

        search_res = result.value
        
        # Redaction logic
        final_matches = []
        for match in search_res.matches:
            ip_display = match.ip_str
            if not args.no_redact:
                ip_display = redact_ip(ip_display)
            
            # Create a dict representation for JSON output or display
            # We reconstruct it to handle redaction easily
            m_dict = {
                "ip_str": ip_display,
                "port": match.port,
                "org": match.org,
                "location": {
                    "country_name": match.location.country_name,
                    "city": match.location.city
                },
                "hostnames": match.hostnames,
                "product": match.product,
                **match.metadata
            }
            final_matches.append(m_dict)

        if args.json:
            print(json.dumps({"total": search_res.total, "matches": final_matches}, indent=2))
        else:
            print(f"Total results: {search_res.total}")
            print(f"Showing: {len(final_matches)}")
            print()
            for m in final_matches:
                print(f"IP: {m['ip_str']}")
                print(f"  Port: {m['port']}")
                print(f"  Org: {m['org']}")
                print(f"  Product: {m['product']}")
                print(f"  Country: {m['location']['country_name']}")
                print(f"  Hostnames: {', '.join(m['hostnames']) if m['hostnames'] else 'N/A'}")
                print()
            if not args.no_redact:
                print("Note: IP addresses partially redacted per security policy")

    elif args.command == "host":
        result = asyncio.run(client.host(args.ip))

        if result.is_err():
            print(f"Error: {result.error.message}", file=sys.stderr)
            sys.exit(1)

        host_res = result.value
        display_ip = host_res.ip_str
        if not args.no_redact:
            display_ip = redact_ip(display_ip)

        # Build dict
        h_dict = {
            "ip_str": display_ip,
            "org": host_res.org,
            "isp": host_res.isp,
            "asn": host_res.asn,
            "country_name": host_res.country_name,
            "city": host_res.city,
            "ports": host_res.ports,
            "hostnames": host_res.hostnames,
            "tags": host_res.tags,
            **host_res.metadata
        }

        if args.json:
            print(json.dumps(h_dict, indent=2))
        else:
            print(f"IP: {display_ip}")
            print(f"Org: {h_dict['org']}")
            print(f"ISP: {h_dict['isp']}")
            print(f"ASN: {h_dict['asn']}")
            print(f"Country: {h_dict['country_name']}")
            print(f"City: {h_dict['city']}")
            print(f"Ports: {', '.join(map(str, h_dict['ports'])) if h_dict['ports'] else 'N/A'}")
            print(f"Hostnames: {', '.join(h_dict['hostnames']) if h_dict['hostnames'] else 'N/A'}")
            print(f"Tags: {', '.join(h_dict['tags']) if h_dict['tags'] else 'N/A'}")

    elif args.command == "dns":
        hostnames = [h.strip() for h in args.hostname.split(",")]
        result = asyncio.run(client.dns_resolve(hostnames))

        if result.is_err():
            print(f"Error: {result.error.message}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result.value, indent=2))
        else:
            for hostname, ip in result.value.items():
                print(f"{hostname}: {ip}")

    elif args.command == "reverse":
        ips = [ip.strip() for ip in args.ip.split(",")]
        result = asyncio.run(client.dns_reverse(ips))

        if result.is_err():
            print(f"Error: {result.error.message}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result.value, indent=2))
        else:
            for ip, hostnames in result.value.items():
                hosts = ", ".join(hostnames) if hostnames else "No hostnames found"
                print(f"{ip}: {hosts}")

    elif args.command == "cve":
        result = asyncio.run(client.cve(args.cve_id.upper()))

        if result.is_err():
            print(f"Error: {result.error.message}", file=sys.stderr)
            sys.exit(1)

        cve = result.value
        # Build dict for display
        cve_dict = {
            "cve_id": cve.cve_id,
            "summary": cve.summary,
            "cvss_v3": cve.cvss_v3,
            "cvss_v2": cve.cvss_v2,
            "epss": cve.epss,
            "kev": cve.kev,
            "propose_action": cve.propose_action,
            "ransomware_campaign": cve.ransomware_campaign,
            "published_time": cve.published_time,
            "references": cve.references
        }

        if args.json:
            print(json.dumps(cve_dict, indent=2))
        else:
            print(f"CVE ID: {cve.cve_id}")
            print(f"Summary: {cve.summary or 'N/A'}")
            print()
            print("Severity:")
            if cve.cvss_v3:
                print(f"  CVSS v3: {cve.cvss_v3} ({format_cve_severity(cve.cvss_v3)})")
            if cve.cvss_v2:
                print(f"  CVSS v2: {cve.cvss_v2} ({format_cve_severity(cve.cvss_v2)})")
            if cve.epss:
                print(f"  EPSS: {cve.epss * 100:.2f}%")
            print()
            print(f"KEV (Known Exploited): {'Yes' if cve.kev else 'No'}")
            if cve.propose_action:
                print(f"Proposed Action: {cve.propose_action}")
            if cve.ransomware_campaign:
                print(f"Ransomware Campaign: {cve.ransomware_campaign}")
            print()
            print(f"Published: {cve.published_time or 'N/A'}")
            if cve.references:
                print()
                print("References:")
                for ref in cve.references[:5]:
                    print(f"  - {ref}")

if __name__ == "__main__":
    main()

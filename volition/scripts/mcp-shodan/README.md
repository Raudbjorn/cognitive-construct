# Shodan API

Security reconnaissance API wrapper for Shodan.

## Usage

```bash
python shodan_api.py search "apache port:443" --confirm
python shodan_api.py host 8.8.8.8
python shodan_api.py dns google.com
python shodan_api.py reverse 8.8.8.8
python shodan_api.py cve CVE-2021-44228
```

## Environment Variables

- `SHODAN_API_KEY`: Required Shodan API key

## Security Features (R.22-R.23)

- **Confirmation required**: Search queries require `--confirm` flag
- **IP redaction**: IP addresses are partially redacted by default (use `--no-redact` to disable)
- **Audit logging**: All queries are logged for audit purposes

## Commands

| Command | Description |
|---------|-------------|
| `search` | Search Shodan database (requires --confirm) |
| `host` | Get information about a specific IP |
| `dns` | Resolve hostname to IP address |
| `reverse` | Reverse DNS lookup |
| `cve` | Look up CVE information from CVEDB |

## Dependencies

```bash
pip install httpx
```

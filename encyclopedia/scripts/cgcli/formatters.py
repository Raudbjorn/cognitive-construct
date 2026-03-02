"""Output formatters: table, json, toon."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any, Sequence


def detect_format(explicit: str | None = None) -> str:
    """Auto-detect output format: table for TTY, toon for piped output."""
    if explicit:
        return explicit
    if sys.stdout.isatty():
        return "table"
    return "toon"


def format_output(
    data: Any,
    fmt: str,
    columns: list[str] | None = None,
) -> str:
    """Format data according to the specified format."""
    if fmt == "json":
        return format_json(data)
    elif fmt == "toon":
        return format_toon(data)
    else:
        return format_table(data, columns)


def format_json(data: Any) -> str:
    if isinstance(data, list):
        items = [asdict(d) if hasattr(d, '__dataclass_fields__') else d for d in data]
        return json.dumps(items, indent=2, default=str)
    if hasattr(data, '__dataclass_fields__'):
        return json.dumps(asdict(data), indent=2, default=str)
    return json.dumps(data, indent=2, default=str)


def format_toon(data: Any) -> str:
    try:
        from py_toon_format import encode

        if isinstance(data, list):
            items = [asdict(d) if hasattr(d, '__dataclass_fields__') else d for d in data]
            return encode(items)
        if hasattr(data, '__dataclass_fields__'):
            return encode(asdict(data))
        return encode(data)
    except ImportError:
        # Fallback to json if toon not installed
        return format_json(data)


def format_table(data: Any, columns: list[str] | None = None) -> str:
    """Column-aligned table output. Uses rich if installed, else plain."""
    if not data:
        return "(no results)"

    items: list[dict] = []
    if isinstance(data, list):
        for d in data:
            items.append(asdict(d) if hasattr(d, '__dataclass_fields__') else d)
    elif hasattr(data, '__dataclass_fields__'):
        items = [asdict(data)]
    else:
        return str(data)

    if not items:
        return "(no results)"

    cols = columns or list(items[0].keys())
    # Filter out None-heavy columns
    cols = [c for c in cols if any(item.get(c) is not None for item in items)]

    try:
        return _rich_table(items, cols)
    except ImportError:
        return _plain_table(items, cols)


def _rich_table(items: list[dict], cols: list[str]) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    for col in cols:
        table.add_column(col)

    for item in items:
        table.add_row(*[_truncate(str(item.get(col, "")), 80) for col in cols])

    console = Console(width=200)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _plain_table(items: list[dict], cols: list[str]) -> str:
    widths = {col: len(col) for col in cols}
    for item in items:
        for col in cols:
            val = _truncate(str(item.get(col, "")), 60)
            widths[col] = max(widths[col], len(val))

    lines: list[str] = []
    header = "  ".join(col.ljust(widths[col]) for col in cols)
    lines.append(header)
    lines.append("  ".join("-" * widths[col] for col in cols))

    for item in items:
        row = "  ".join(
            _truncate(str(item.get(col, "")), 60).ljust(widths[col])
            for col in cols
        )
        lines.append(row)

    return "\n".join(lines)


def _truncate(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s

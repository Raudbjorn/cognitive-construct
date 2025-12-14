"""Storage utilities for vibe log persistence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import CATEGORY_KEYWORDS, DATA_DIR, LOG_FILE
from .types import CategorySummary, LearningEntry, LearningType


def _ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _empty_log() -> dict[str, Any]:
    """Create empty log structure."""
    return {"mistakes": {}, "last_updated": int(time.time() * 1000)}


def read_log_file() -> dict[str, Any]:
    """Read the vibe log from disk."""
    _ensure_data_dir()

    if not LOG_FILE.exists():
        log = _empty_log()
        write_log_file(log)
        return log

    try:
        data = LOG_FILE.read_text(encoding="utf-8")
        return json.loads(data)
    except (json.JSONDecodeError, OSError):
        return _empty_log()


def write_log_file(data: dict[str, Any]) -> None:
    """Write data to the vibe log file."""
    _ensure_data_dir()

    try:
        json_data = json.dumps(data, indent=2)
        LOG_FILE.write_text(json_data, encoding="utf-8")
    except OSError as e:
        print(f"Error writing vibe log: {e}")


def add_learning_entry(
    mistake: str,
    category: str,
    solution: str | None = None,
    entry_type: LearningType = LearningType.MISTAKE,
) -> LearningEntry:
    """Add a learning entry to the vibe log."""
    log = read_log_file()
    now = int(time.time() * 1000)

    entry = LearningEntry(
        type=entry_type,
        category=category,
        mistake=mistake,
        solution=solution,
        timestamp=now,
    )

    # Initialize category if it doesn't exist
    if category not in log["mistakes"]:
        log["mistakes"][category] = {
            "count": 0,
            "examples": [],
            "lastUpdated": now,
        }

    # Update category data
    log["mistakes"][category]["count"] += 1
    log["mistakes"][category]["examples"].append({
        "type": entry_type.value,
        "category": category,
        "mistake": mistake,
        "solution": solution,
        "timestamp": now,
    })
    log["mistakes"][category]["lastUpdated"] = now
    log["last_updated"] = now

    write_log_file(log)
    return entry


def get_learning_entries() -> dict[str, list[LearningEntry]]:
    """Get all learning entries grouped by category."""
    log = read_log_file()
    result: dict[str, list[LearningEntry]] = {}

    for category, data in log.get("mistakes", {}).items():
        entries = []
        for ex in data.get("examples", []):
            entries.append(
                LearningEntry(
                    type=LearningType(ex.get("type", "mistake")),
                    category=ex.get("category", category),
                    mistake=ex.get("mistake", ""),
                    solution=ex.get("solution"),
                    timestamp=ex.get("timestamp", 0),
                )
            )
        result[category] = entries

    return result


def get_learning_category_summary() -> list[CategorySummary]:
    """Get learning category summaries, sorted by count."""
    log = read_log_file()
    summary: list[CategorySummary] = []

    for category, data in log.get("mistakes", {}).items():
        examples = data.get("examples", [])
        if not examples:
            continue

        recent = examples[-1]
        recent_entry = LearningEntry(
            type=LearningType(recent.get("type", "mistake")),
            category=recent.get("category", category),
            mistake=recent.get("mistake", ""),
            solution=recent.get("solution"),
            timestamp=recent.get("timestamp", 0),
        )

        summary.append(
            CategorySummary(
                category=category,
                count=data.get("count", len(examples)),
                recent_example=recent_entry,
            )
        )

    # Sort by count descending
    return sorted(summary, key=lambda x: x.count, reverse=True)


def get_learning_context_text(max_per_category: int = 5) -> str:
    """Build a learning context string from the vibe log."""
    log = read_log_file()
    context_parts: list[str] = []

    for category, data in log.get("mistakes", {}).items():
        context_parts.append(f"Category: {category} (count: {data.get('count', 0)})")

        examples = sorted(
            data.get("examples", []),
            key=lambda x: x.get("timestamp", 0),
        )[-max_per_category:]

        for ex in examples:
            from datetime import datetime, timezone

            ts = ex.get("timestamp", 0)
            date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()

            entry_type = ex.get("type", "mistake")
            label = {
                "mistake": "Mistake",
                "preference": "Preference",
                "success": "Success",
            }.get(entry_type, "Mistake")

            solution_text = f" | Solution: {ex['solution']}" if ex.get("solution") else ""
            context_parts.append(f"- [{date}] {label}: {ex.get('mistake', '')}{solution_text}")

        context_parts.append("")

    return "\n".join(context_parts).strip()


def normalize_category(category: str) -> str:
    """Normalize category to a standard category if possible."""
    lower_category = category.lower()

    for standard_category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lower_category for keyword in keywords):
            return standard_category

    return category


def enforce_one_sentence(text: str) -> str:
    """Ensure text is a single sentence."""
    # Remove newlines
    sentence = text.replace("\r\n", " ").replace("\n", " ")

    # Split by sentence-ending punctuation
    import re

    parts = re.split(r"([.!?])\s+", sentence)

    # Take first sentence with punctuation
    if len(parts) > 1:
        sentence = parts[0] + (parts[1] if len(parts) > 1 else "")
    sentence = sentence.strip()

    # Ensure ends with punctuation
    if not re.search(r"[.!?]$", sentence):
        sentence += "."

    return sentence


def is_similar(a: str, b: str) -> bool:
    """Simple similarity check between two sentences."""
    import re

    a_words = [w for w in re.split(r"\W+", a.lower()) if w]
    b_words = [w for w in re.split(r"\W+", b.lower()) if w]

    if not a_words or not b_words:
        return False

    overlap = [w for w in a_words if w in b_words]
    ratio = len(overlap) / min(len(a_words), len(b_words))
    return ratio >= 0.6

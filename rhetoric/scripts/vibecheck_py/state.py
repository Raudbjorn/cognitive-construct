"""Session state management for vibe check history."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import DATA_DIR, HISTORY_FILE
from .types import Interaction, VibeCheckInput


_history: dict[str, list[dict[str, Any]]] = {}
_initialized = False


def _ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def load_history() -> None:
    """Load history from disk."""
    global _history, _initialized

    _ensure_data_dir()

    try:
        if HISTORY_FILE.exists():
            data = HISTORY_FILE.read_text(encoding="utf-8")
            _history = json.loads(data)
        else:
            _history = {"default": []}
    except (json.JSONDecodeError, OSError):
        _history = {"default": []}

    _initialized = True


def load_history_sync() -> None:
    """Load history from disk (sync version)."""
    global _history, _initialized

    _ensure_data_dir()

    try:
        if HISTORY_FILE.exists():
            data = HISTORY_FILE.read_text(encoding="utf-8")
            _history = json.loads(data)
        else:
            _history = {"default": []}
    except (json.JSONDecodeError, OSError):
        _history = {"default": []}

    _initialized = True


def _save_history() -> None:
    """Save history to disk."""
    _ensure_data_dir()
    try:
        json_data = json.dumps(_history)
        HISTORY_FILE.write_text(json_data, encoding="utf-8")
    except OSError:
        pass


def get_history_summary(session_id: str = "default") -> str:
    """Get a summary of recent history for context."""
    if not _initialized:
        load_history_sync()

    sess_history = _history.get(session_id, [])
    if not sess_history:
        return ""

    # Take last 5 interactions
    recent = sess_history[-5:]
    summary_parts: list[str] = []

    for i, interaction in enumerate(recent, 1):
        input_data = interaction.get("input", {})
        output = interaction.get("output", "")
        goal = input_data.get("goal", "Unknown")
        guidance = output[:100] if output else ""
        summary_parts.append(f"Interaction {i}: Goal {goal}, Guidance: {guidance}...")

    return f"History Context:\n" + "\n".join(summary_parts) + "\n"


def add_to_history(
    session_id: str | None,
    input_data: VibeCheckInput,
    output: str,
) -> None:
    """Add an interaction to history."""
    if not _initialized:
        load_history_sync()

    sid = session_id or "default"

    if sid not in _history:
        _history[sid] = []

    # Convert input to dict for storage
    input_dict = {
        "goal": input_data.goal,
        "plan": input_data.plan,
        "user_prompt": input_data.user_prompt,
        "progress": input_data.progress,
        "uncertainties": list(input_data.uncertainties),
        "task_context": input_data.task_context,
        "session_id": input_data.session_id,
    }

    _history[sid].append({
        "input": input_dict,
        "output": output,
        "timestamp": int(time.time() * 1000),
    })

    # Keep only last 10
    if len(_history[sid]) > 10:
        _history[sid] = _history[sid][-10:]

    _save_history()


def is_initialized() -> bool:
    """Check if history has been initialized."""
    return _initialized


def ensure_initialized() -> None:
    """Ensure history is initialized."""
    if not _initialized:
        load_history_sync()

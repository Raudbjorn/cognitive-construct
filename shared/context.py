import os
import tomllib
from typing import Dict, Any
from pathlib import Path

AI_DIR = Path(os.path.expanduser("~/.ai"))
MANIFEST_PATH = AI_DIR / "developer-memory-manifest.toml"
ASSISTANT_MANIFEST_PATH = AI_DIR / "assistant_manifest.json"

def load_manifest() -> Dict[str, Any]:
    """
    Loads the developer memory manifest from ~/.ai/developer-memory-manifest.toml
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)

def get_system_prompt_additions() -> str:
    """
    Generates text to append to system prompts based on the manifest.
    """
    try:
        data = load_manifest()
        sections = ["\n## SHARED AI CONTEXT (from ~/.ai)"]

        if "assistant" in data:
            if "focus" in data["assistant"]:
                sections.append(f"### Focus Areas:\n- " + "\n- ".join(data['assistant']['focus']))
            if "workflow_priorities" in data["assistant"]:
                sections.append(f"\n### Workflow Priorities:\n" + "\n".join(data['assistant']['workflow_priorities']))

        if "guidelines" in data:
             for group, details in data["guidelines"].items():
                 if "rules" in details:
                     sections.append(f"\n### {group.replace('_', ' ').title()} Guidelines:")
                     for rule in details["rules"]:
                         sections.append(f"- {rule}")

        # Load Assistant Manifest (JSON)
        if ASSISTANT_MANIFEST_PATH.exists():
            import json
            try:
                with open(ASSISTANT_MANIFEST_PATH, "r") as f:
                    adata = json.load(f)

                if "technologies" in adata:
                    sections.append("\n### Technology Specific Rules:")
                    for tech in adata["technologies"]:
                        sections.append(f"\n#### {tech['name']} ({tech['category']})")
                        for feature in tech.get("features", []):
                            sections.append(f"\n**{feature['name']}**:")
                            if "rules" in feature:
                                for rule in feature["rules"]:
                                    sections.append(f"- [{rule.get('severity', 'INFO')}] {rule.get('description')}")
                                    if "correct_pattern" in rule:
                                        # sanitize newlines for prompt
                                        pat = rule['correct_pattern'].replace('\n', ' ')
                                        sections.append(f"  - Correct: `{pat}`")
            except Exception as e:
                 sections.append(f"\nError loading assistant manifest: {e}")

        return "\n".join(sections)
    except Exception as e:
        return f"Error loading context: {e}"

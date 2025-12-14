"""
Credential validation and management for Claude Skills.

Implements D.23: Centralize credential loading/masking/routing.
Patterns validated during Phase 0.1 (2025-12-13).
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

from .errors import SkillError, ErrorCode


class CredentialValidator:
    """
    Validates API key formats based on known provider patterns.
    Updated with patterns from Phase 0.1 credential inventory (2025-12-13).
    """

    PATTERNS = {
        # LLM Providers (11 keys, 10 unique - GROK duplicates XAI)
        "ANTHROPIC_API_KEY": r"^sk-ant-[a-zA-Z0-9_\-]+$",
        "OPENAI_API_KEY": r"^sk-(proj-)?[a-zA-Z0-9_\-]+$",
        "OPENROUTER_API_KEY": r"^sk-or-v1-[a-f0-9]+$",
        "GOOGLE_CLOUD_API_KEY": r"^AIzaSy[a-zA-Z0-9_\-]+$",
        "DEEPSEEK_API_KEY": r"^sk-[a-f0-9]+$",
        "MISTRAL_API_KEY": r"^[a-zA-Z0-9]+$",
        "XAI_API_KEY": r"^xai-[a-zA-Z0-9]+$",
        "GROK_API_KEY": r"^xai-[a-zA-Z0-9]+$",  # Duplicate of XAI
        "TOGETHER_API_KEY": r"^tgp_v1_[a-zA-Z0-9_\-]+$",
        "PERPLEXITY_API_KEY": r"^pplx-[a-zA-Z0-9_\-]+$",
        "KIMI_API_KEY": r"^sk-[a-zA-Z0-9]+$",

        # Search/Knowledge (6 keys)
        "CONTEXT7_API_KEY": r"^ctx7sk-[a-f0-9\-]+$",
        "EXA_API_KEY": r"^[a-f0-9\-]{36}$",  # UUID format
        "KAGI_API_KEY": r"^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+$",  # JWT-like
        "BRAVE_API_KEY": r"^BSA[a-zA-Z0-9]+$",
        "SERPER_API_KEY": r"^[a-f0-9]{40}$",
        "TAVILY_API_KEY": r"^tvly-(dev-)?[a-zA-Z0-9]+$",

        # Memory/Storage (5 keys)
        "MEM0_API_KEY": r"^m0-[a-zA-Z0-9]+$",
        "OPENMEMORY_API_KEY": r"^om-[a-z0-9]+$",
        "QDRANT_API_KEY": r"^[a-f0-9\-]+\|[a-zA-Z0-9_\-]+$",  # UUID|token
        "ZEP_API_KEY": r"^z_[a-zA-Z0-9_\.\-]+$",
        "ZERO_ENTROPY_API_KEY": r"^ze_[a-zA-Z0-9]+$",

        # Media/Audio (3 keys)
        "ELEVENLABS_API_KEY": r"^sk_[a-f0-9]+$",
        "LIVEKIT_API_KEY": r"^API[a-zA-Z0-9]+$",
        "LIVEKIT_API_SECRET": r"^[a-zA-Z0-9]+$",

        # Observability (4 keys)
        "AGENTOPS_API_KEY": r"^[a-f0-9\-]{36}$",  # UUID
        "HELICONE_API_KEY": r"^sk-helicone-[a-z0-9\-]+$",
        "KEYWORDSAI_API_KEY": r"^lka[a-zA-Z0-9]+\.[a-zA-Z0-9]+$",
        "SEGMENT_API_KEY": r"^[a-zA-Z0-9]+$",

        # Infrastructure (3 keys)
        "AZURE_AI_SEARCH_API_KEY": r"^[a-zA-Z0-9]+$",
        "SARVAM_API_KEY": r"^sk_[a-z0-9_]+$",
        "GOOGLE_CLOUD_PROJECT": r"^[a-z0-9\-]+$",
    }

    # Credentials known to be non-working (Phase 0.2 validation)
    NON_WORKING = {
        "XAI_API_KEY": "Team has no credits - purchase at console.x.ai",
        "GROK_API_KEY": "Team has no credits - same as XAI_API_KEY",
        "KIMI_API_KEY": "Invalid authentication - re-generate at Moonshot console",
        "KAGI_API_KEY": "Closed beta - email support@kagi.com for access",
    }

    # Skill to credential mapping
    SKILL_CREDENTIALS = {
        "encyclopedia": [
            "EXA_API_KEY",  # Required
            "CONTEXT7_API_KEY",  # Recommended
            "PERPLEXITY_API_KEY",  # Recommended
            "BRAVE_API_KEY",  # Optional
            "SERPER_API_KEY",  # Optional
            "TAVILY_API_KEY",  # Optional
        ],
        "rhetoric": [
            # At least 2 LLM providers needed for deliberation
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_CLOUD_API_KEY",
            "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
            "PERPLEXITY_API_KEY",
        ],
        "inland-empire": [
            # mcp-memory-libsql works without API key
            "MEM0_API_KEY",  # Optional (hosted mode)
            "QDRANT_API_KEY",  # Optional
            "ZEP_API_KEY",  # Optional
        ],
        "volition": [
            "OPENAI_API_KEY",  # For websearch
            "ELEVENLABS_API_KEY",  # Optional (TTS)
            "LIVEKIT_API_KEY",  # Optional (real-time)
            "LIVEKIT_API_SECRET",  # Optional (real-time)
        ],
    }

    @classmethod
    def validate(cls, key_name: str, key_value: str) -> bool:
        """
        Validates a key against its known pattern.
        Returns True if valid or if no pattern exists (fail open for unknown keys).
        """
        if not key_value:
            return False

        pattern = cls.PATTERNS.get(key_name)
        if pattern:
            if not re.match(pattern, key_value):
                return False

        # Basic sanity check for all keys
        if len(key_value.strip()) < 8:
            return False

        return True

    @classmethod
    def check_environment(cls, required_keys: List[str]) -> None:
        """
        Checks if required keys are present and valid in os.environ.
        Raises SkillError if missing or invalid.
        """
        missing = []
        invalid = []

        for key in required_keys:
            val = os.environ.get(key)
            if not val:
                missing.append(key)
                continue

            if not cls.validate(key, val):
                invalid.append(key)

        if missing or invalid:
            error_msg = ""
            if missing:
                error_msg += f"Missing credentials: {', '.join(missing)}. "
            if invalid:
                error_msg += f"Invalid format for: {', '.join(invalid)}. "

            raise SkillError(ErrorCode.SECURITY_ERROR, error_msg.strip())

    @classmethod
    def get_available_credentials(cls) -> Dict[str, bool]:
        """
        Returns a dict of all known credentials and whether they are available.
        Does not reveal credential values.
        """
        result = {}
        for key in cls.PATTERNS:
            val = os.environ.get(key)
            result[key] = val is not None and cls.validate(key, val)
        return result

    @classmethod
    def get_skill_credentials(cls, skill: str) -> Dict[str, bool]:
        """
        Returns availability of credentials for a specific skill.
        """
        keys = cls.SKILL_CREDENTIALS.get(skill, [])
        return {key: os.environ.get(key) is not None for key in keys}

    @classmethod
    def is_known_non_working(cls, key_name: str) -> Optional[str]:
        """
        Returns the reason if a credential is known to be non-working.
        """
        return cls.NON_WORKING.get(key_name)


def mask_credential(val: str, show_last: int = 4) -> str:
    """Masks a credential for display."""
    if not val or len(val) <= show_last:
        return "****"
    return "*" * (len(val) - show_last) + val[-show_last:]


def get_credential_inventory() -> Dict[str, Dict]:
    """
    Returns a comprehensive inventory of all credentials.
    For runtime metrics and task exposure (D.23).
    """
    inventory = {}
    for key, pattern in CredentialValidator.PATTERNS.items():
        val = os.environ.get(key)
        inventory[key] = {
            "present": val is not None,
            "valid": CredentialValidator.validate(key, val) if val else False,
            "masked": mask_credential(val) if val else None,
            "non_working_reason": CredentialValidator.is_known_non_working(key),
        }
    return inventory


# ---------------------------------------------------------------------------
# Credential Router (central inventory + loader)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILES: Tuple[Path, ...] = (
    PROJECT_ROOT / ".env.local",
    PROJECT_ROOT / ".env",
)


def _load_env_files(paths: Iterable[Path]) -> None:
    """Load key/value pairs from env files without overriding existing variables."""
    for path in paths:
        if not path or not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class SkillCredentialSpec:
    """Registration details per skill."""
    skill: str
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)


@dataclass
class CredentialStatus:
    """Inventory record for a single credential."""
    key: str
    present: bool
    optional: bool
    masked_value: Optional[str] = None


class CredentialRouter:
    """
    Centralize credential loading/masking/routing (addresses D.23 / T.21).
    """

    _instance: Optional["CredentialRouter"] = None

    def __new__(cls, env_files: Iterable[Path] = DEFAULT_ENV_FILES):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(env_files)
        return cls._instance

    def _init(self, env_files: Iterable[Path]) -> None:
        _load_env_files(env_files)
        self._skills: Dict[str, SkillCredentialSpec] = {}
        self._inventory: Dict[str, List[CredentialStatus]] = {}

    # ------------------------------------------------------------------
    # Registration + Lookup
    # ------------------------------------------------------------------
    def register_skill(
        self,
        skill: str,
        required: Optional[List[str]] = None,
        optional: Optional[List[str]] = None,
    ) -> None:
        """Register a skill's credential needs."""
        spec = SkillCredentialSpec(
            skill=skill,
            required=required or [],
            optional=optional or [],
        )
        self._skills[skill] = spec
        self._inventory[skill] = self._build_inventory(spec)

    def get_credentials(self, skill: str) -> Dict[str, str]:
        """Return credentials for a skill, raising if required keys missing."""
        spec = self._skills.get(skill)
        if not spec:
            raise SkillError(ErrorCode.SECURITY_ERROR, f"Skill '{skill}' is not registered with CredentialRouter")

        missing = [key for key in spec.required if not os.environ.get(key)]
        if missing:
            raise SkillError(
                ErrorCode.SECURITY_ERROR,
                f"Missing credentials for {skill}: {', '.join(missing)}"
            )

        resolved: Dict[str, str] = {}
        for key in spec.required + spec.optional:
            value = os.environ.get(key)
            if value:
                resolved[key] = value
        return resolved

    # ------------------------------------------------------------------
    # Inventory + Metrics
    # ------------------------------------------------------------------
    def _build_inventory(self, spec: SkillCredentialSpec) -> List[CredentialStatus]:
        statuses: List[CredentialStatus] = []
        for key in spec.required + spec.optional:
            value = os.environ.get(key)
            statuses.append(
                CredentialStatus(
                    key=key,
                    present=bool(value),
                    optional=key in spec.optional,
                    masked_value=mask_credential(value) if value else None,
                )
            )
        return statuses

    def get_inventory(self, skill: Optional[str] = None) -> Dict[str, List[CredentialStatus]]:
        """Return safe inventory metadata for one or all skills."""
        if skill:
            spec = self._skills.get(skill)
            if not spec:
                raise SkillError(ErrorCode.SECURITY_ERROR, f"Skill '{skill}' is not registered")
            self._inventory[skill] = self._build_inventory(spec)
            return {skill: list(self._inventory[skill])}

        for name, spec in self._skills.items():
            self._inventory[name] = self._build_inventory(spec)
        return {name: list(statuses) for name, statuses in self._inventory.items()}

    def inventory_summary(self) -> Dict[str, Dict[str, int]]:
        """Aggregated counts for runtime metrics."""
        summary: Dict[str, Dict[str, int]] = {}
        for skill, statuses in self.get_inventory().items():
            present = sum(1 for status in statuses if status.present)
            missing = sum(1 for status in statuses if not status.present and not status.optional)
            optional_missing = sum(1 for status in statuses if not status.present and status.optional)
            summary[skill] = {
                "present": present,
                "missing": missing,
                "optional_missing": optional_missing,
                "total": len(statuses),
            }
        return summary


def get_credential_router() -> CredentialRouter:
    """Singleton accessor for credential router."""
    return CredentialRouter()

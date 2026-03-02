"""Data-driven language parser system.

Public API:
    get_config(name) -> LanguageConfig
    GenericExtractor
    LanguageConfig
"""

from ._base import GenericExtractor, LanguageConfig
from ._configs import CONFIGS, get_config

__all__ = ["GenericExtractor", "LanguageConfig", "CONFIGS", "get_config"]

"""Test configuration for shared modules.

Pre-imports the modules under test by loading them directly from
their file paths, avoiding the shared/__init__.py which eagerly
imports heavy dependencies (mcp, model2vec, etc.).
"""

import importlib.util
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).parent.parent

# Register individual modules into sys.modules so tests can do
# `from shared.vocabulary import Vocabulary` etc. without triggering
# the shared package __init__.py dependency chain.

_MODULES_TO_PRELOAD = ["vocabulary", "query_pipeline", "fusion"]

for mod_name in _MODULES_TO_PRELOAD:
    full_name = f"shared.{mod_name}"
    if full_name in sys.modules:
        continue

    # Ensure `shared` exists as a namespace in sys.modules
    if "shared" not in sys.modules:
        import types
        pkg = types.ModuleType("shared")
        pkg.__path__ = [str(SHARED_DIR)]
        pkg.__package__ = "shared"
        sys.modules["shared"] = pkg

    mod_path = SHARED_DIR / f"{mod_name}.py"
    if not mod_path.exists():
        continue

    spec = importlib.util.spec_from_file_location(full_name, mod_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        # Also set as attribute on the package
        setattr(sys.modules["shared"], mod_name, mod)

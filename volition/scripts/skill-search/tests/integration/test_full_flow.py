"""Integration tests for full frontend → backend flow."""

import subprocess
import time
import pytest


def test_frontend_startup_time():
    """Test that frontend starts quickly (critical for Cursor timeout)."""
    # Use frontend from published PyPI (once published)
    # For now, test the import speed only
    from pathlib import Path
    
    # Find frontend directory relative to this test file
    test_dir = Path(__file__).parent
    frontend_dir = test_dir.parent.parent / "packages" / "frontend"
    
    start_time = time.time()
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from skill_search import __version__; print(__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=str(frontend_dir),
    )
    elapsed = time.time() - start_time

    print(f"\n✓ Frontend import time: {elapsed:.2f}s")
    assert result.returncode == 0, f"Frontend failed: {result.stderr}"
    assert elapsed < 2, f"Frontend import took {elapsed:.2f}s (should be instant)"
    print(f"✓ Frontend version: {result.stdout.strip()}")


def test_backend_available_from_pypi():
    """Test that backend is available and downloadable from PyPI."""
    result = subprocess.run(
        ["uvx", "skill-search-backend", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Backend not available: {result.stderr}"
    assert "skill-search-backend" in result.stdout.lower()
    assert "--port" in result.stdout
    assert "--host" in result.stdout
    print("\n✓ Backend available from PyPI")


def test_backend_starts_http_server():
    """Test that backend starts and responds to health checks."""

    # Start backend in background
    proc = subprocess.Popen(
        ["uvx", "skill-search-backend", "--port", "9999", "--verbose"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for startup
        time.sleep(10)

        # Test health endpoint
        response = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:9999/health"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        print(f"\n✓ Backend health response: {response.stdout}")
        assert "ok" in response.stdout.lower()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()  # Force kill if doesn't terminate
            proc.wait()


def test_hardcoded_tools_match_backend():
    """Verify frontend's hardcoded tool schemas match backend."""
    from skill_search.mcp_proxy import TOOL_SCHEMAS

    # Check we have the right number of tools
    assert len(TOOL_SCHEMAS) == 3, f"Expected 3 tools, got {len(TOOL_SCHEMAS)}"

    # Check tool names
    tool_names = [tool.name for tool in TOOL_SCHEMAS]
    assert "find_helpful_skills" in tool_names
    assert "read_skill_document" in tool_names
    assert "list_skills" in tool_names

    # Check schemas have required fields
    for tool in TOOL_SCHEMAS:
        assert tool.name
        assert tool.description
        assert tool.inputSchema
        print(f"✓ Tool schema valid: {tool.name}")


def test_frontend_backend_schemas_identical():
    """Verify frontend and backend tool schemas are identical.

    This is critical - schema mismatches cause 'invalid tool use' errors in Claude.
    Both packages now import from their own tool_schemas.py which should be identical.

    Uses order-independent comparison to avoid false failures on reordering.
    """
    from skill_search.tool_schemas import (
        TOOL_SCHEMAS as FRONTEND_SCHEMAS,
        DEFAULT_TOP_K as FRONTEND_DEFAULT_TOP_K,
    )
    from skill_search_backend.tool_schemas import (
        TOOL_SCHEMAS as BACKEND_SCHEMAS,
        DEFAULT_TOP_K as BACKEND_DEFAULT_TOP_K,
    )

    # Check constants match
    assert FRONTEND_DEFAULT_TOP_K == BACKEND_DEFAULT_TOP_K, (
        f"DEFAULT_TOP_K mismatch: frontend={FRONTEND_DEFAULT_TOP_K}, "
        f"backend={BACKEND_DEFAULT_TOP_K}"
    )
    print(f"✓ DEFAULT_TOP_K matches: {FRONTEND_DEFAULT_TOP_K}")

    # Build maps keyed by tool name (order-independent)
    frontend_by_name = {tool.name: tool for tool in FRONTEND_SCHEMAS}
    backend_by_name = {tool.name: tool for tool in BACKEND_SCHEMAS}

    # Ensure both sides expose the same tool set
    assert frontend_by_name.keys() == backend_by_name.keys(), (
        f"Tool set mismatch: "
        f"frontend-only={frontend_by_name.keys() - backend_by_name.keys()}, "
        f"backend-only={backend_by_name.keys() - frontend_by_name.keys()}"
    )
    print(f"✓ Same tool set: {sorted(frontend_by_name.keys())}")

    # Enforce full schema equality per tool (order-independent)
    # Uses model_dump() to compare ALL fields, catching any divergence
    for name in sorted(frontend_by_name.keys()):
        frontend_tool = frontend_by_name[name]
        backend_tool = backend_by_name[name]

        # Full comparison via model_dump() catches any field divergence
        assert frontend_tool.model_dump() == backend_tool.model_dump(), (
            f"Tool schema mismatch for {name}"
        )
        print(f"✓ Schema identical: {name}")


@pytest.mark.asyncio
async def test_backend_manager_check():
    """Test that backend manager can check for uvx."""
    from skill_search.backend_manager import BackendManager

    manager = BackendManager()
    has_uvx = manager.check_backend_available()

    assert has_uvx, "uvx not available - required for backend spawning"
    print("\n✓ Backend manager can check uvx availability")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

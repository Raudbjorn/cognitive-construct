"""Semgrep client implementation using Result types."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semgrep_mcp.result import Err, Ok, Result
from semgrep_mcp.types import CodeFile, SemgrepError, SemgrepScanResult

# === Constants ===

DEFAULT_TIMEOUT_SECONDS = 120.0
SEMGREP_PATH_ENV_VAR = "SEMGREP_PATH"

# Common locations where semgrep might be installed
COMMON_SEMGREP_PATHS = [
    "semgrep",
    "/usr/local/bin/semgrep",
    "/usr/bin/semgrep",
    "/opt/homebrew/bin/semgrep",
    "/opt/semgrep/bin/semgrep",
    "/home/linuxbrew/.linuxbrew/bin/semgrep",
    "/snap/bin/semgrep",
]


# === Helper Functions ===


def _find_semgrep() -> str | None:
    """Find semgrep executable in PATH or common locations."""
    # Check environment variable first
    env_path = os.environ.get(SEMGREP_PATH_ENV_VAR)
    if env_path and os.path.isfile(env_path):
        return env_path

    # Try shutil.which first (fastest)
    result = shutil.which("semgrep")
    if result:
        return result

    # Check common locations
    for path in COMMON_SEMGREP_PATHS:
        if os.path.isabs(path) and os.path.isfile(path):
            return path

    return None


def _get_semgrep_version(semgrep_path: str) -> Result[str, SemgrepError]:
    """Get semgrep version."""
    try:
        result = subprocess.run(
            [semgrep_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Ok(result.stdout.strip())
        return Err(SemgrepError(message=f"Version check failed: {result.stderr}"))
    except subprocess.TimeoutExpired:
        return Err(SemgrepError(message="Version check timed out", code="TIMEOUT"))
    except OSError as e:
        return Err(SemgrepError(message=f"Failed to run semgrep: {e}", code="OS_ERROR"))


def _safe_join(base_dir: str, untrusted_path: str) -> Result[str, SemgrepError]:
    """Safely join base directory with untrusted path, preventing path traversal."""
    base_path = Path(base_dir).resolve()

    if not untrusted_path or untrusted_path == "." or untrusted_path.strip("/") == "":
        return Ok(base_path.as_posix())

    if Path(untrusted_path).is_absolute():
        return Err(SemgrepError(message="Path must be relative", code="INVALID_PATH"))

    full_path = base_path / Path(untrusted_path)

    if full_path != full_path.resolve():
        return Err(
            SemgrepError(
                message=f"Path escapes base directory: {untrusted_path}",
                code="PATH_TRAVERSAL",
            )
        )

    return Ok(full_path.as_posix())


def _parse_scan_output(output: str) -> Result[dict[str, Any], SemgrepError]:
    """Parse JSON output from semgrep scan."""
    try:
        return Ok(json.loads(output))
    except json.JSONDecodeError as e:
        return Err(SemgrepError(message=f"Invalid JSON output: {e}", code="PARSE_ERROR"))


def _clean_temp_paths(scan_result: SemgrepScanResult, temp_dir: str) -> SemgrepScanResult:
    """Clean up temporary directory paths in scan results."""
    import contextlib

    cleaned_results = []
    for finding in scan_result.results:
        if "path" in finding:
            with contextlib.suppress(ValueError):
                finding["path"] = os.path.relpath(finding["path"], temp_dir)
        cleaned_results.append(finding)

    cleaned_paths = dict(scan_result.paths)
    if "scanned" in cleaned_paths:
        cleaned_paths["scanned"] = [os.path.relpath(p, temp_dir) for p in cleaned_paths["scanned"]]

    return SemgrepScanResult(
        version=scan_result.version,
        results=cleaned_results,
        paths=cleaned_paths,
        errors=list(scan_result.errors),
        skipped_rules=list(scan_result.skipped_rules),
    )


def _create_temp_file(
    temp_dir: str, code_file: CodeFile
) -> Result[None, SemgrepError]:
    """Create a temporary file from CodeFile."""
    if not code_file.path:
        return Ok(None)

    join_result = _safe_join(temp_dir, code_file.path)
    if join_result.is_err():
        return join_result  # type: ignore[return-value]

    temp_file_path = join_result.value

    try:
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "w") as f:
            f.write(code_file.content)
        return Ok(None)
    except OSError as e:
        return Err(
            SemgrepError(
                message=f"Failed to create temp file {code_file.path}: {e}",
                code="IO_ERROR",
            )
        )


# === Client Class ===


@dataclass
class SemgrepClient:
    """Client for Semgrep CLI operations.

    Usage:
        client = SemgrepClient()

        # Scan local files
        result = await client.scan_files(["/path/to/code"])
        if result.is_ok():
            for finding in result.value.results:
                print(f"{finding['path']}: {finding['check_id']}")
        else:
            print(f"Error: {result.error.message}")

        # Scan code content directly
        files = [CodeFile(path="example.py", content="import os\\nos.system(input())")]
        result = await client.scan_content(files, config="p/security")
    """

    timeout: float = DEFAULT_TIMEOUT_SECONDS
    semgrep_path: str | None = None
    _resolved_path: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize resolved semgrep path."""
        self._resolved_path = self.semgrep_path or _find_semgrep()

    def _check_semgrep(self) -> Result[str, SemgrepError]:
        """Verify semgrep is available."""
        if not self._resolved_path:
            return Err(
                SemgrepError(
                    message=(
                        "Semgrep not found. Install with: pip install semgrep, "
                        "brew install semgrep (macOS), or see https://semgrep.dev/docs/getting-started/"
                    ),
                    code="NOT_FOUND",
                )
            )
        return Ok(self._resolved_path)

    async def _run_semgrep(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> Result[str, SemgrepError]:
        """Run semgrep command and return stdout."""
        check_result = self._check_semgrep()
        if check_result.is_err():
            return check_result  # type: ignore[return-value]

        semgrep_path = check_result.value
        full_args = [semgrep_path, *args]

        run_env = {**os.environ, "SEMGREP_MCP": "true"}
        if env:
            run_env.update(env)

        try:
            process = await asyncio.create_subprocess_exec(
                *full_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=run_env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return Err(
                    SemgrepError(
                        message=f"Semgrep timed out after {self.timeout}s",
                        code="TIMEOUT",
                    )
                )

            if process.returncode != 0 and not stdout:
                return Err(
                    SemgrepError(
                        message=stderr.decode() if stderr else f"Exit code {process.returncode}",
                        code="EXECUTION_ERROR",
                        status_code=process.returncode,
                    )
                )

            return Ok(stdout.decode())

        except OSError as e:
            return Err(SemgrepError(message=f"Failed to run semgrep: {e}", code="OS_ERROR"))

    # === Public Methods ===

    async def get_version(self) -> Result[str, SemgrepError]:
        """Get semgrep version.

        Returns:
            Result containing version string on success or SemgrepError on failure
        """
        check_result = self._check_semgrep()
        if check_result.is_err():
            return check_result  # type: ignore[return-value]

        return _get_semgrep_version(check_result.value)

    async def get_supported_languages(self) -> Result[list[str], SemgrepError]:
        """Get list of languages supported by semgrep.

        Returns:
            Result containing list of language names on success or SemgrepError on failure
        """
        result = await self._run_semgrep(["show", "supported-languages"])

        if result.is_err():
            return result  # type: ignore[return-value]

        languages = [line.strip() for line in result.value.split("\n") if line.strip()]
        return Ok(languages)

    async def scan_files(
        self,
        paths: list[str],
        config: str | None = None,
        experimental: bool = True,
    ) -> Result[SemgrepScanResult, SemgrepError]:
        """Scan local files with semgrep.

        Args:
            paths: List of file or directory paths to scan
            config: Semgrep config (e.g., "auto", "p/security", path to rules file)
            experimental: Use experimental mode to avoid extra exec (default: True)

        Returns:
            Result containing SemgrepScanResult on success or SemgrepError on failure
        """
        # Validate paths exist
        for path in paths:
            if not os.path.exists(path):
                return Err(
                    SemgrepError(
                        message=f"Path not found: {path}",
                        code="PATH_NOT_FOUND",
                    )
                )

        args = ["scan", "--json"]
        if experimental:
            args.append("--experimental")
        if config:
            args.extend(["--config", config])
        args.extend(paths)

        result = await self._run_semgrep(args)
        if result.is_err():
            return result  # type: ignore[return-value]

        parse_result = _parse_scan_output(result.value)
        if parse_result.is_err():
            return parse_result  # type: ignore[return-value]

        data = parse_result.value
        return Ok(SemgrepScanResult.from_dict(data))

    async def scan_content(
        self,
        files: list[CodeFile],
        config: str | None = None,
        experimental: bool = True,
    ) -> Result[SemgrepScanResult, SemgrepError]:
        """Scan code content (not files) with semgrep.

        Creates temporary files from the provided content, scans them,
        and cleans up afterwards.

        Args:
            files: List of CodeFile objects with path and content
            config: Semgrep config (e.g., "auto", "p/security", path to rules file)
            experimental: Use experimental mode to avoid extra exec (default: True)

        Returns:
            Result containing SemgrepScanResult on success or SemgrepError on failure
        """
        if not files:
            return Err(SemgrepError(message="No files provided to scan", code="INVALID_INPUT"))

        temp_dir = tempfile.mkdtemp(prefix="semgrep_scan_")

        try:
            # Create temporary files
            for code_file in files:
                create_result = _create_temp_file(temp_dir, code_file)
                if create_result.is_err():
                    return create_result  # type: ignore[return-value]

            # Scan the temp directory
            result = await self.scan_files([temp_dir], config=config, experimental=experimental)

            if result.is_ok():
                return Ok(_clean_temp_paths(result.value, temp_dir))

            return result

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def validate_rule(self, rule_yaml: str) -> Result[bool, SemgrepError]:
        """Validate a semgrep rule YAML.

        Args:
            rule_yaml: YAML content of the rule to validate

        Returns:
            Result containing True if valid, or SemgrepError with validation errors
        """
        temp_dir = tempfile.mkdtemp(prefix="semgrep_rule_")
        rule_file = os.path.join(temp_dir, "rule.yaml")

        try:
            with open(rule_file, "w") as f:
                f.write(rule_yaml)

            result = await self._run_semgrep(["--validate", "--config", rule_file])

            if result.is_err():
                return Err(
                    SemgrepError(
                        message=f"Rule validation failed: {result.error.message}",
                        code="VALIDATION_ERROR",
                    )
                )

            return Ok(True)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# === Convenience Functions ===


async def scan_files(
    paths: list[str],
    config: str | None = None,
    semgrep_path: str | None = None,
) -> Result[SemgrepScanResult, SemgrepError]:
    """Scan files with semgrep (convenience function).

    Creates a client for single use. For multiple requests,
    prefer creating a SemgrepClient instance.
    """
    client = SemgrepClient(semgrep_path=semgrep_path)
    return await client.scan_files(paths, config)


async def scan_content(
    files: list[CodeFile],
    config: str | None = None,
    semgrep_path: str | None = None,
) -> Result[SemgrepScanResult, SemgrepError]:
    """Scan code content with semgrep (convenience function).

    Creates a client for single use. For multiple requests,
    prefer creating a SemgrepClient instance.
    """
    client = SemgrepClient(semgrep_path=semgrep_path)
    return await client.scan_content(files, config)


async def get_version(
    semgrep_path: str | None = None,
) -> Result[str, SemgrepError]:
    """Get semgrep version (convenience function)."""
    client = SemgrepClient(semgrep_path=semgrep_path)
    return await client.get_version()

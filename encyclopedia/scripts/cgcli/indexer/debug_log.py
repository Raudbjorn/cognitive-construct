"""Logging shim matching CGC's debug_log API.

Delegates to stdlib logging — no config_manager dependency.
"""

import logging

log = logging.getLogger(__name__)


def debug_log(message: str) -> None:
    """Write debug message (file-based in CGC, stdlib here)."""
    log.debug(message)


def info_logger(msg: str) -> None:
    log.info(msg)


def error_logger(msg: str) -> None:
    log.error(msg)


def warning_logger(msg: str) -> None:
    log.warning(msg)


def debug_logger(msg: str) -> None:
    log.debug(msg)

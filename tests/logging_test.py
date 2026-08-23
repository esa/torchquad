"""Regression tests for loguru library hygiene (issue #184).

torchquad must not reconfigure loguru at import time or wipe a host
application's log handlers when the user sets the torchquad log level.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from torchquad import MonteCarlo, set_log_level

# Importing torchquad configures logging exactly once, so the import-time
# behaviour of TORCHQUAD_LOG_LEVEL can only be observed in a fresh interpreter.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMPORT_SNIPPET = "import torchquad"


def _import_torchquad_with_env(log_level):
    """Import torchquad in a fresh interpreter and return what it wrote to stderr.

    Args:
        log_level (str or None): Value for TORCHQUAD_LOG_LEVEL, or None to leave
            the variable unset.

    Returns:
        str: The subprocess's stderr.
    """
    env = dict(os.environ)
    env.pop("TORCHQUAD_LOG_LEVEL", None)
    if log_level is not None:
        env["TORCHQUAD_LOG_LEVEL"] = log_level

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_SNIPPET],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stderr


def test_set_log_level_preserves_host_handlers():
    """set_log_level must not remove handlers registered by the host application.

    Previously set_log_level called logger.remove() with no argument, which
    removed every sink including the host's (issue #184).
    """
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="INFO", filter=lambda record: True)
    try:
        set_log_level("WARNING")
        logger.info("message from the host application")
        assert "message from the host application" in host_sink.getvalue(), (
            "set_log_level removed the host application's loguru handler"
        )
    finally:
        logger.remove(host_handler_id)
        # Remove the handlers set_log_level added and restore the library default
        # so this test does not leak an enabled state or stderr sink into later tests.
        import torchquad.utils.set_log_level as set_log_level_module

        for handler_id in set_log_level_module._torchquad_handler_ids:
            logger.remove(handler_id)
        set_log_level_module._torchquad_handler_ids.clear()
        logger.disable("torchquad")


def test_torchquad_records_disabled_by_default():
    """Importing torchquad disables its own loguru records, so running an
    integration must not leak torchquad log output into a host handler."""
    # Assert the library default explicitly for order-independence (other tests
    # may have called set_log_level, which enables torchquad's records).
    logger.disable("torchquad")
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="DEBUG", filter=lambda record: True)
    try:
        MonteCarlo().integrate(
            lambda x: x, dim=1, N=100, integration_domain=[[0.0, 1.0]], backend="numpy"
        )
        assert "Computed integral" not in host_sink.getvalue(), (
            "torchquad emitted log records into the host handler while disabled"
        )
    finally:
        logger.remove(host_handler_id)
        logger.disable("torchquad")


def test_import_is_silent_without_log_level_env_var():
    """A bare import must print nothing, so torchquad stays silent in any host app."""
    stderr = _import_torchquad_with_env(None)
    assert stderr == "", f"importing torchquad wrote to stderr: {stderr!r}"


def test_log_level_env_var_enables_logging_at_import():
    """TORCHQUAD_LOG_LEVEL must switch logging on without editing the source.

    It replaces the old TORCHQUAD_DISABLE_LOGGING constant, which could only be
    changed by editing __init__.py and therefore never took effect for users.
    """
    stderr = _import_torchquad_with_env("DEBUG")
    assert "Initializing torchquad." in stderr, (
        f"TORCHQUAD_LOG_LEVEL=DEBUG did not enable logging; stderr was {stderr!r}"
    )


def test_torchquad_sink_honors_the_configured_level():
    """The handler torchquad adds must respect the level it was given.

    Only torchquad's own sink is asserted here, identified by its "TQ-" format
    marker. Once records are enabled they also reach any other sink loguru has
    registered -- including loguru's own default stderr handler, which is at
    DEBUG and unfiltered -- and a library cannot lower the level of a handler it
    does not own without touching the host's configuration.
    """
    stderr = _import_torchquad_with_env("ERROR")
    assert "TQ-" not in stderr, (
        f"torchquad's own sink emitted below its ERROR level; stderr was {stderr!r}"
    )

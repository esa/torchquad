"""Regression tests for loguru library hygiene (issue #184).

torchquad must not reconfigure loguru at import time or wipe a host
application's log handlers when the user sets the torchquad log level.
"""

import io
import os
import subprocess
import sys

import pytest
from loguru import logger

from torchquad import MonteCarlo, set_log_level

# Importing torchquad configures logging exactly once, so the import-time
# behaviour of TORCHQUAD_LOG_LEVEL can only be observed in a fresh interpreter.

# Records emitted by torchquad's own sink carry this marker in their format.
# Asserting on it is what separates torchquad's handler from loguru's stock one,
# which is unfiltered, sits at DEBUG, and prints every enabled record as well.
_SINK_MARKER = "TQ-"

# Emit one record above the configured level and one below it, as torchquad
# itself would. loguru decides both enable/disable and a sink's filter from the
# calling frame's __name__, so the probe has to genuinely live in the torchquad
# namespace: it is built as a module named "torchquad.probe" and the records are
# emitted from inside it. logger.patch() is not a substitute -- it rewrites the
# record's name only after the enable/disable check has already run, so a patched
# probe keeps logging even while torchquad is disabled and would make the
# silent-by-default test pass for the wrong reason.
_PROBE_SNIPPET = """
import sys
import types

import torchquad

probe = types.ModuleType("torchquad.probe")
sys.modules["torchquad.probe"] = probe
exec(
    "from loguru import logger\\n"
    "def emit():\\n"
    "    logger.error('PROBE-ERROR')\\n"
    "    logger.info('PROBE-INFO')\\n",
    probe.__dict__,
)
probe.emit()
"""


def _run_with_env(snippet, log_level):
    """Run a snippet in a fresh interpreter and return what it wrote to stderr.

    Args:
        snippet (str): Python source to execute.
        log_level (str or None): Value for TORCHQUAD_LOG_LEVEL, or None to leave
            the variable unset.

    Returns:
        str: The subprocess's stderr.
    """
    env = dict(os.environ)
    env.pop("TORCHQUAD_LOG_LEVEL", None)
    if log_level is not None:
        env["TORCHQUAD_LOG_LEVEL"] = log_level

    # No cwd override: the subprocess must import the installed torchquad, the
    # same one every other test file uses. Running from the repository root
    # would put the source tree ahead of site-packages and quietly test
    # something else.
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stderr


def _sink_lines(stderr):
    """Return only the lines torchquad's own handler produced.

    Args:
        stderr (str): Captured stderr from a subprocess.

    Returns:
        list: Lines carrying torchquad's sink marker.
    """
    return [line for line in stderr.splitlines() if _SINK_MARKER in line]


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
    """A bare import must print nothing, so torchquad stays silent in any host app.

    The probe snippet is used rather than a plain import so that this also covers
    records emitted after import, not just the initialization message.

    Only torchquad's own output is asserted on. Requiring stderr to be empty
    outright would make this hostage to any warning a transitive import happens
    to emit.
    """
    stderr = _run_with_env(_PROBE_SNIPPET, None)

    assert _SINK_MARKER not in stderr, f"torchquad added a sink while silent: {stderr!r}"
    for leaked in ("Initializing torchquad.", "PROBE-ERROR", "PROBE-INFO"):
        assert leaked not in stderr, (
            f"torchquad emitted {leaked!r} while silent; stderr was {stderr!r}"
        )


@pytest.mark.parametrize("log_level", ["", None])
def test_unset_or_empty_log_level_leaves_the_library_silent(log_level):
    """An empty TORCHQUAD_LOG_LEVEL must mean "off", not crash the import.

    `TORCHQUAD_LOG_LEVEL=` is the ordinary shell and CI idiom for an unset
    variable. Passing that straight to loguru raises `ValueError: Level '' does
    not exist` from inside the import, taking the whole library down.
    """
    stderr = _run_with_env(_PROBE_SNIPPET, log_level)

    assert _SINK_MARKER not in stderr, f"torchquad added a sink while silent: {stderr!r}"


def test_invalid_log_level_fails_loudly():
    """A non-empty but unknown level must raise rather than be silently ignored.

    The message has to name the valid levels: this surfaces as a failed
    ``import torchquad``, where loguru's bare "Level 'X' does not exist" gives
    the reader nothing to act on.
    """
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        _run_with_env("import torchquad", "NOT_A_LEVEL")

    stderr = excinfo.value.stderr
    assert "Unknown log level 'NOT_A_LEVEL'" in stderr, (
        f"expected a level error naming the bad value, got {stderr!r}"
    )
    assert "WARNING" in stderr, f"the error should list the valid levels, got {stderr!r}"


def test_log_level_env_var_adds_torchquads_own_sink():
    """TORCHQUAD_LOG_LEVEL must switch logging on without editing the source.

    It replaces the old TORCHQUAD_DISABLE_LOGGING constant, which could only be
    changed by editing __init__.py and so never took effect for users.

    The assertion is on torchquad's own marked output. Merely finding the
    initialization message anywhere in stderr would not distinguish this from
    loguru's stock handler printing it, which happens whether or not torchquad
    adds a sink of its own.
    """
    stderr = _run_with_env("import torchquad", "DEBUG")
    marked = _sink_lines(stderr)

    assert any("Initializing torchquad." in line for line in marked), (
        f"torchquad's own sink did not emit the initialization record; its lines "
        f"were {marked!r} out of {stderr!r}"
    )


def test_torchquad_sink_honors_the_configured_level():
    """torchquad's handler must pass records at its level and drop those below.

    Both halves matter. Asserting only that nothing appears below the level is
    vacuous -- it holds equally when no sink was added at all -- so a record
    above the level has to be shown getting through the same sink.
    """
    stderr = _run_with_env(_PROBE_SNIPPET, "ERROR")
    marked = _sink_lines(stderr)

    assert any("PROBE-ERROR" in line for line in marked), (
        f"torchquad's sink dropped a record at its own ERROR level; its lines "
        f"were {marked!r} out of {stderr!r}"
    )
    assert not any("PROBE-INFO" in line for line in marked), (
        f"torchquad's sink emitted an INFO record while set to ERROR; its lines were {marked!r}"
    )


def test_records_still_reach_handlers_torchquad_does_not_own():
    """Enabled records reach every registered sink, not only torchquad's.

    This is the documented limitation behind the level, and the reason the README
    cannot promise that a level filters everything: a library cannot lower the
    level of a handler it did not add without touching the host's configuration,
    which is exactly what issue #184 was about. Pinning the behaviour here means
    what the docs describe is what is tested, and that changing it later is a
    deliberate act rather than an accident.
    """
    stderr = _run_with_env(_PROBE_SNIPPET, "ERROR")

    assert "PROBE-INFO" in stderr, (
        "expected the INFO probe to reach loguru's own default handler, which is "
        f"unfiltered and sits at DEBUG; stderr was {stderr!r}"
    )
    info_lines = [line for line in stderr.splitlines() if "PROBE-INFO" in line]
    assert not any(_SINK_MARKER in line for line in info_lines), (
        f"the INFO probe must reach the default handler only, not torchquad's "
        f"sink; matching lines were {info_lines!r}"
    )

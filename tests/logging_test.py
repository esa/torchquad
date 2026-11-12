"""Regression tests for loguru library hygiene (issue #184).

torchquad must stay silent until asked, must never touch a host application's
log handlers, and must honour the level the user sets on every handler.
"""

import io
import math
import os
import subprocess
import sys

import pytest
from loguru import logger

import torchquad.utils.library_logger as library_logger
from torchquad import MonteCarlo, set_log_level

# Importing torchquad configures logging exactly once, so the import-time
# behaviour of TORCHQUAD_LOG_LEVEL can only be observed in a fresh interpreter.
# A fresh interpreter also has loguru's stock stderr handler, unfiltered and at
# DEBUG, which is exactly the handler a level set by torchquad used to miss.

# One eager integration. It logs "Computed integral" at INFO and several
# records, e.g. "Checking inputs to Integrator.", at DEBUG.
_INTEGRATION_SNIPPET = """
import numpy as np

import torchquad

torchquad.Trapezoid().integrate(
    lambda points: points[:, 0], dim=1, N=11, integration_domain=np.array([[0.0, 1.0]])
)
"""

# The same integration through TensorFlow's compiled path. AutoGraph re-executes
# torchquad's code under a tensorflow module name, which a name-based
# logger.disable("torchquad") does not cover.
_TENSORFLOW_COMPILED_SNIPPET = """
import tensorflow as tf

import torchquad

torchquad.set_up_backend("tensorflow", "float64")
domain = tf.constant([[0.0, 1.0]], dtype=tf.float64)
compiled = torchquad.Trapezoid().get_jit_compiled_integrate(
    dim=1, N=11, integration_domain=domain
)
compiled(lambda points: points[:, 0], domain)
"""

_INFO_MESSAGE = "Computed integral"
_DEBUG_MESSAGE = "Checking inputs to Integrator."


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


@pytest.fixture
def silent_library():
    """Put torchquad back into its silent default before and after a test.

    Other test modules call set_log_level, and both the enabled state and the
    level gate are process-wide.
    """
    logger.disable("torchquad")
    library_logger._LevelGate.minimum_level_number = math.inf
    yield
    logger.disable("torchquad")
    library_logger._LevelGate.minimum_level_number = math.inf


def test_set_log_level_preserves_host_handlers(silent_library):
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


def test_torchquad_records_disabled_by_default(silent_library):
    """Without set_log_level an integration must not reach a host handler."""
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="DEBUG", filter=lambda record: True)
    try:
        MonteCarlo().integrate(
            lambda x: x, dim=1, N=100, integration_domain=[[0.0, 1.0]], backend="numpy"
        )
        assert host_sink.getvalue() == "", (
            f"torchquad emitted log records while silent: {host_sink.getvalue()!r}"
        )
    finally:
        logger.remove(host_handler_id)


def test_level_applies_to_handlers_torchquad_does_not_own(silent_library):
    """The level must hold on a host handler that itself accepts everything.

    A handler's own level cannot do this: the host's handler here sits at
    DEBUG, like loguru's default one, so only a gate at the source keeps
    torchquad's DEBUG records out of it. Each admitted record must also arrive
    once, which it would not if torchquad registered a second sink.
    """
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="DEBUG", filter=lambda record: True)
    try:
        set_log_level("INFO")
        MonteCarlo().integrate(
            lambda x: x, dim=1, N=100, integration_domain=[[0.0, 1.0]], backend="numpy"
        )
        output = host_sink.getvalue()
        assert output.count(_INFO_MESSAGE) == 1, f"expected one INFO record, got {output!r}"
        assert "DEBUG" not in output, f"a DEBUG record passed an INFO level: {output!r}"
        # The record must be attributed to the module that logged it, not to the
        # gate, or the host could no longer filter torchquad's records by name.
        assert "torchquad.integration.monte_carlo" in output
    finally:
        logger.remove(host_handler_id)


def test_import_is_silent_without_log_level_env_var():
    """A bare import and an integration must print no torchquad record.

    Only torchquad's own output is asserted on. Requiring stderr to be empty
    outright would make this hostage to any warning a transitive import happens
    to emit.
    """
    stderr = _run_with_env(_INTEGRATION_SNIPPET, None)

    for leaked in (_INFO_MESSAGE, _DEBUG_MESSAGE, "torchquad."):
        assert leaked not in stderr, (
            f"torchquad emitted {leaked!r} while silent; stderr was {stderr!r}"
        )


def test_empty_log_level_leaves_the_library_silent():
    """An empty TORCHQUAD_LOG_LEVEL must mean "off", not crash the import.

    `TORCHQUAD_LOG_LEVEL=` is the ordinary shell and CI idiom for an unset
    variable, so it must not be passed on as a level name.
    """
    stderr = _run_with_env(_INTEGRATION_SNIPPET, "")

    assert _INFO_MESSAGE not in stderr, f"torchquad logged with an empty level: {stderr!r}"


def test_invalid_log_level_fails_loudly():
    """A non-empty but unknown level must raise rather than be silently ignored.

    The message has to name the valid levels: this surfaces as a failed
    ``import torchquad``, where a bare "level does not exist" gives the reader
    nothing to act on.
    """
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        _run_with_env("import torchquad", "NOT_A_LEVEL")

    stderr = excinfo.value.stderr
    assert "Unknown log level 'NOT_A_LEVEL'" in stderr, (
        f"expected a level error naming the bad value, got {stderr!r}"
    )
    assert "WARNING" in stderr, f"the error should list the valid levels, got {stderr!r}"


def test_log_level_env_var_switches_logging_on():
    """TORCHQUAD_LOG_LEVEL must switch logging on without editing the source."""
    stderr = _run_with_env(_INTEGRATION_SNIPPET, "DEBUG")

    assert _INFO_MESSAGE in stderr, f"no INFO record at level DEBUG: {stderr!r}"
    assert _DEBUG_MESSAGE in stderr, f"no DEBUG record at level DEBUG: {stderr!r}"


def test_level_is_honoured_by_the_default_handler():
    """In a plain interpreter a level must show each admitted record exactly once.

    loguru's default handler is unfiltered and at DEBUG. With the level applied
    only to a torchquad-owned sink, it printed every DEBUG record regardless and
    every admitted record a second time.
    """
    at_info = _run_with_env(_INTEGRATION_SNIPPET, "INFO")
    assert at_info.count(_INFO_MESSAGE) == 1, f"expected one INFO record, got {at_info!r}"
    assert _DEBUG_MESSAGE not in at_info, f"a DEBUG record passed an INFO level: {at_info!r}"

    at_warning = _run_with_env(_INTEGRATION_SNIPPET, "WARNING")
    assert _INFO_MESSAGE not in at_warning, f"an INFO record passed a WARNING level: {at_warning!r}"


def test_tensorflow_compiled_path_is_silent_by_default():
    """The compiled TensorFlow path must not log while logging is off.

    AutoGraph runs torchquad's code under a tensorflow module name, so these
    records were not covered by logger.disable("torchquad") and reached stderr.
    """
    pytest.importorskip("tensorflow")
    stderr = _run_with_env(_TENSORFLOW_COMPILED_SNIPPET, None)

    for leaked in (_INFO_MESSAGE, "Creating a grid", "Integration grid created"):
        assert leaked not in stderr, (
            f"torchquad emitted {leaked!r} from the compiled TensorFlow path while "
            f"silent; stderr was {stderr!r}"
        )

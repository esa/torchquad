"""Tests for the benchmark harness's timing helpers.

The harness previously stopped its clock before any result was materialized. On
an asynchronous GPU backend that measures kernel-launch latency instead of the
integration, understating runtime by more than an order of magnitude at large N
and inflating every speedup computed against it.

The central test therefore has to be able to *fail* if materialization leaves
the timed region. Asserting that a real workload takes more than zero seconds
cannot do that -- `perf_counter` always advances, so the broken implementation
passes too. Instead a stand-in result is used whose materialization takes a
known, deliberately slow amount of time, which makes the measured interval
depend on whether materialization happened inside the clock or outside it. That
works on CPU-only runners as well, and costs milliseconds rather than a
multi-million-element workload.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarking"))

from timing import materialize, synchronize, time_integration  # noqa: E402

# Long enough to dwarf the timer's resolution and any incidental work in
# time_integration, short enough not to slow the suite down.
_MATERIALIZE_DELAY_SECONDS = 0.25


class _SlowToMaterialize:
    """A stand-in for a lazy backend result whose materialization is slow.

    Attributes:
        materialized (bool): Whether ``item`` has been called.
    """

    def __init__(self):
        self.materialized = False

    def item(self):
        """Return the value, taking a known amount of time to do so.

        Returns:
            float: The result value.
        """
        time.sleep(_MATERIALIZE_DELAY_SECONDS)
        self.materialized = True
        return 2.5


def test_materialize_accepts_a_plain_float():
    assert materialize(2.5) == 2.5


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "tensorflow"])
def test_materialize_returns_a_python_float_for_every_backend(backend):
    """A backend tensor must come back as a float, not a tensor.

    Returning a tensor would let a caller keep the computation lazy and so defeat
    the point of materializing inside the timed region. Every backend is
    exercised rather than skipped silently: CI installs all four and runs with
    --error-for-skips, so a missing one is a failure, not a quiet gap.
    """
    module = pytest.importorskip(backend)

    if backend == "numpy":
        value = module.float64(2.5)
    elif backend == "torch":
        value = module.tensor(2.5)
    elif backend == "jax":
        value = module.numpy.array(2.5)
    else:
        value = module.constant(2.5)

    result = materialize(value)
    assert type(result) is float, f"{backend} materialized to {type(result)}, not float"
    assert result == pytest.approx(2.5), f"{backend} materialized to the wrong value"


def test_synchronize_is_a_no_op_without_cuda():
    """It must be safe to call on any machine, including CPU-only CI runners."""
    synchronize()


def test_time_integration_materializes_inside_the_measured_region():
    """The measured interval must include the cost of materializing the result.

    This is the regression that matters: if materialization moves back outside
    the clock, the elapsed time collapses to the cost of returning the lazy
    handle and this assertion fails.
    """
    result = _SlowToMaterialize()
    elapsed, value = time_integration(lambda: result)

    assert result.materialized, "time_integration returned without materializing the result"
    assert value == pytest.approx(2.5)
    assert elapsed >= _MATERIALIZE_DELAY_SECONDS, (
        f"materialization was not inside the timed region: measured {elapsed:.4f}s "
        f"for work that cannot take less than {_MATERIALIZE_DELAY_SECONDS}s"
    )


def test_timing_without_materialization_would_miss_the_work():
    """Control for the test above: the old measurement really was blind to this.

    Without this, an assertion that a slow operation takes at least its own
    duration would be unfalsifiable -- it would hold for any implementation that
    happened to be slow for other reasons.
    """
    result = _SlowToMaterialize()

    # The shape the harness used to have: run the call, stop the clock, and only
    # then force the value.
    start = time.perf_counter()
    produced = (lambda: result)()
    unmeasured = time.perf_counter() - start
    materialize(produced)

    assert result.materialized, "the control never materialized, so it proves nothing"
    assert unmeasured < _MATERIALIZE_DELAY_SECONDS, (
        "the control is not measuring what it claims: stopping the clock before "
        f"materializing still took {unmeasured:.4f}s"
    )

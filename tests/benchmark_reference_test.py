"""Tests that the benchmark harness never invents a ground truth.

Every error on every convergence plot is measured against the reference value,
so substituting one when the calculation fails does not produce a visibly broken
benchmark -- it produces a plausible one that is quietly wrong for every method
at once. That is the same class of defect as the GPU timing bug, and harder to
notice, because nothing in the output looks unusual.

The harness used to return ``1.0`` in that case.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarking"))

from modular_benchmark import ModularBenchmark  # noqa: E402


class _HarnessStub:
    """Minimal stand-in providing what ``get_reference_value`` reads.

    Calling the method unbound on this avoids ``ModularBenchmark.__init__``,
    which configures CUDA, precision and file logging -- none of which this
    behaviour depends on.
    """

    def __init__(self):
        self.logger = logging.getLogger("benchmark_reference_test")
        self.config = {"convergence": {}}


def _failing_integrand(points):
    """An integrand that always raises, to force the numerical fallback to fail.

    Args:
        points: Ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("integrand deliberately failed")


def test_analytical_reference_is_returned_for_known_dimensions():
    """The tabulated references must still be used; this is not a blanket raise."""
    value = ModularBenchmark.get_reference_value(_HarnessStub(), None, 1, [[0.0, 1.0]])
    assert value == pytest.approx(4.0422850545e-01)


def test_failed_reference_raises_instead_of_substituting_a_value():
    """A reference that cannot be computed must abort, not become a number.

    Asserting on the type alone would not distinguish this from the integrand's
    own error escaping, so the message is checked too: it has to say which
    dimension failed and why that matters.
    """
    with pytest.raises(RuntimeError) as excinfo:
        ModularBenchmark.get_reference_value(
            _HarnessStub(), _failing_integrand, 2, [[0.0, 1.0]] * 2
        )

    message = str(excinfo.value)
    assert "reference value for 2D" in message, message
    assert "meaningless" in message, message
    # The original failure has to survive, or the report is unactionable.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "integrand deliberately failed" in str(excinfo.value.__cause__)


def test_the_old_fallback_value_is_gone():
    """Guard the specific regression: 1.0 must not come back as a reference.

    A future refactor could reintroduce a default without failing the test above
    if it raised for some other reason, so pin the value itself.
    """
    with pytest.raises(RuntimeError):
        result = ModularBenchmark.get_reference_value(
            _HarnessStub(), _failing_integrand, 2, [[0.0, 1.0]] * 2
        )
        assert result != 1.0, "the fabricated fallback reference is back"

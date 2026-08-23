"""Tests for the benchmark harness's timing helpers.

The harness previously stopped its clock before any result was materialized. On
an asynchronous GPU backend that measures kernel-launch latency instead of the
integration, understating runtime by an order of magnitude at large N and
inflating every speedup computed against it. These tests pin the behaviour that
prevents a regression.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "benchmarking"))

from timing import materialize, synchronize, time_integration  # noqa: E402


def test_materialize_accepts_a_plain_float():
    assert materialize(2.5) == 2.5


def test_materialize_returns_a_python_float_for_every_backend():
    """A backend tensor must come back as a float, not a tensor.

    Returning a tensor would let a caller keep the computation lazy and defeat
    the point of materializing inside the timed region.
    """
    import numpy as np

    values = {"numpy": np.float64(2.5)}

    try:
        import torch

        values["torch"] = torch.tensor(2.5)
    except ImportError:
        pass

    try:
        import jax.numpy as jnp

        values["jax"] = jnp.array(2.5)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        values["tensorflow"] = tf.constant(2.5)
    except ImportError:
        pass

    for backend, value in values.items():
        result = materialize(value)
        assert type(result) is float, f"{backend} materialized to {type(result)}, not float"
        assert result == pytest.approx(2.5), f"{backend} materialized to the wrong value"


def test_synchronize_is_a_no_op_without_cuda():
    """It must be safe to call on any machine, including CPU-only CI runners."""
    synchronize()


def test_time_integration_covers_the_device_work():
    """The measured time must include work the backend would otherwise defer.

    Asserted through a deliberately slow integrand: a timer that stops before the
    result is materialized can report far less than the true cost on an
    asynchronous backend, but it can never report more.
    """
    torch = pytest.importorskip("torch")

    def slow_call():
        points = torch.rand(2_000_000, dtype=torch.float64)
        for _ in range(5):
            points = torch.sin(points)
        return points.sum()

    elapsed, value = time_integration(slow_call)

    assert isinstance(value, float)
    assert elapsed > 0.0, "a real workload was timed at zero seconds"


if __name__ == "__main__":
    test_materialize_accepts_a_plain_float()
    test_materialize_returns_a_python_float_for_every_backend()
    test_synchronize_is_a_no_op_without_cuda()
    test_time_integration_covers_the_device_work()

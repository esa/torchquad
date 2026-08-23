"""Timing helpers shared by the benchmark harness.

GPU work is asynchronous. ``integrate()`` returns as soon as its kernels are
queued, long before the device has run them, so a clock stopped at that point
measures kernel-launch latency rather than the integration. The gap grows with
N, since larger kernels take longer to run but no longer to enqueue, and at
large N it is more than an order of magnitude.

Every timed region in this package must therefore end with :func:`materialize`,
which blocks until the device has actually produced the result, and start after
:func:`synchronize`, so that work queued by a previous iteration is not charged
to this one. Comparing a materialized measurement against a non-materialized one
is worse still: the ratio is then a measurement artifact rather than a speedup.
"""

import time


def synchronize():
    """Block until previously queued device work has finished.

    Call this immediately before starting a clock so that setup or teardown from
    an earlier iteration is not attributed to the region being timed. It is a
    no-op unless PyTorch is installed with an initialized CUDA context.
    """
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def materialize(result):
    """Force an integration result to a Python float, blocking until it is ready.

    This is what makes a timed region cover the actual computation on every
    backend: ``.item()`` synchronizes for PyTorch and JAX, ``.numpy()`` does the
    same for TensorFlow, and NumPy results are already materialized.

    Args:
        result: Value returned by ``integrate()``, as a backend tensor or a
            plain number.

    Returns:
        float: The result as a Python float.
    """
    if hasattr(result, "item"):
        return float(result.item())
    if hasattr(result, "numpy"):
        return float(result.numpy())
    return float(result)


def time_integration(integrate_call):
    """Time one integration, charging the device work to the measurement.

    Args:
        integrate_call (callable): Zero-argument callable performing exactly the
            work to be timed, returning the integration result.

    Returns:
        tuple: ``(elapsed_seconds, result_value)`` with the result as a float.
    """
    synchronize()
    start_time = time.perf_counter()
    result = integrate_call()
    result_value = materialize(result)
    elapsed = time.perf_counter() - start_time
    return elapsed, result_value

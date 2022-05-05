import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, to_backend_dtype
import importlib
import pytest
import warnings

from integration.utils import (
    _linspace_with_grads,
    _add_at_indices,
    _setup_integration_domain,
)
from utils.set_precision import set_precision
from utils.enable_cuda import enable_cuda


def _run_tests_with_all_backends(func, func_extra_args=[{}]):
    """Run a test function with all backends and supported precisions

    Args:
        func (function(dtype_name, backend, ...)): Function which runs tests
        func_extra_args (list of dicts, optional): List of extra arguments which are passed to func. Defaults to one execution with no extra arguments.
    """
    # If JAX is tested before Tensorflow here, an out of memory error can
    # happen because both allocate all memory on the GPU by default.
    # The XLA_PYTHON_CLIENT_PREALLOCATE=false environment variable
    # avoids the crash of JAX.
    # For some reason it also does not crash if Tensorflow is tested
    # before JAX, which is done here.
    # Calling block_until_ready on all arrays created with JAX instead
    # of changing the tests order did not avoid the crash for some
    # reason.
    for backend in ["numpy", "torch", "tensorflow", "jax"]:
        if importlib.util.find_spec(backend) is None:
            warnings.warn(f"Backend is not installed: {backend}")
            continue
        if backend == "torch":
            enable_cuda()
        for dtype_name in ["float32", "float64"]:
            set_precision(dtype_name, backend=backend)
            # Iterate over arguments in an inner loop here instead of an outer
            # loop so that there are less switches between backends
            for kwargs in func_extra_args:
                func(dtype_name=dtype_name, backend=backend, **kwargs)


def _run_linspace_with_grads_tests(dtype_name, backend, requires_grad):
    """
    Test _linspace_with_grads with the given dtype, numerical backend and
    requires_grad argument
    """
    if requires_grad and backend != "torch":
        # Currently only torch needs the requires_grad case distinction
        return
    print(
        f"Testing _linspace_with_grads; backend: {backend}, requires_grad:"
        f" {requires_grad}, precision: {dtype_name}"
    )
    dtype_backend = to_backend_dtype(dtype_name, like=backend)
    start = anp.array(-2.0, like=backend, dtype=dtype_backend)
    stop = anp.array(-1.0, like=backend, dtype=dtype_backend)
    assert (
        get_dtype_name(start) == dtype_name
    ), "Unexpected dtype for the configured precision"
    grid1d = _linspace_with_grads(start, stop, 10, requires_grad)
    # Test if the backend, dtype and shape match
    assert infer_backend(grid1d) == backend
    assert grid1d.dtype == start.dtype
    assert grid1d.shape == (10,)
    # The array has to begin at start and end at stop, the elements should
    # be inside [start, stop] and they should be monotonically increasing
    assert grid1d[0] == start and grid1d[9] == stop
    assert all(start <= grid1d[i] <= stop for i in range(10))
    assert all(grid1d[i] < grid1d[i + 1] for i in range(9))


def test_linspace_with_grads():
    """Test _linspace_with_grads with all possible configurations"""
    _run_tests_with_all_backends(
        _run_linspace_with_grads_tests,
        [{"requires_grad": True}, {"requires_grad": False}],
    )


def _run_add_at_indices_tests(dtype_name, backend):
    """
    Test _add_at_indices with the given dtype and numerical backend
    """
    # JAX and Tensorflow are not yet supported
    if backend in ["jax", "tensorflow"]:
        return
    dtype_backend = to_backend_dtype(dtype_name, like=backend)

    print("Testing _add_at_indices for a simple identity case")
    indices = anp.array(list(range(500)), like=backend)
    target = anp.array([0.0] * 500, dtype=dtype_backend, like=backend)
    source = anp.array([1.0] * 500, dtype=dtype_backend, like=backend)
    _add_at_indices(target, indices, source, is_sorted=True)
    assert target.dtype == dtype_backend
    assert target.shape == (500,)
    assert anp.max(anp.abs(target - source)) == 0.0

    print("Testing _add_at_indices when all indices refer to the same target index")
    target = target * 0.0
    indices = indices * 0 + 203
    _add_at_indices(target, indices, source, is_sorted=True)
    assert target[203] == 500.0
    target[203] = 0.0
    assert anp.max(anp.abs(target)) == 0.0

    print("Testing _add_at_indices with unsorted indices and integer dtype")
    target = anp.array([0, 0, 0], like=backend)
    indices = anp.array([2, 1, 1, 2], like=backend)
    source = anp.array([1, 10, 100, 1000], like=backend)
    _add_at_indices(target, indices, source)
    assert target.dtype == indices.dtype
    assert anp.max(anp.abs(target - anp.array([0, 110, 1001], like=backend))) == 0


def test_add_at_indices():
    """Test _add_at_indices with all possible configurations"""
    _run_tests_with_all_backends(_run_add_at_indices_tests)


def _run_setup_integration_domain_tests(dtype_name, backend):
    """
    Test _setup_integration_domain with the given dtype and numerical backend
    """
    # Domain given as List with Python floats
    domain = _setup_integration_domain(2, [[0.0, 1.0], [1.0, 2.0]], backend)
    assert infer_backend(domain) == backend
    assert get_dtype_name(domain) == dtype_name

    # Domain given as List with Python integers
    domain = _setup_integration_domain(2, [[0, 1], [1, 2]], backend)
    assert infer_backend(domain) == backend
    assert get_dtype_name(domain) == dtype_name

    # Domain given as List with mixed precision Python values
    domain = _setup_integration_domain(2, [[0, 1.0], [1, 2.0]], backend)
    assert infer_backend(domain) == backend
    assert get_dtype_name(domain) == dtype_name

    # Default [-1,1]^4 domain
    domain = _setup_integration_domain(4, None, backend)
    assert infer_backend(domain) == backend
    assert get_dtype_name(domain) == dtype_name
    assert domain.shape == (4, 2)

    # User-specified domain
    dtype_backend = to_backend_dtype(dtype_name, like=backend)
    custom_domain = anp.array(
        [[0.0, 1.0], [1.0, 2.0]], like=backend, dtype=dtype_backend
    )
    domain = _setup_integration_domain(2, custom_domain, "unused")
    assert domain.shape == custom_domain.shape
    assert domain.dtype == custom_domain.dtype

    # Tests for invalid arguments
    with pytest.raises(ValueError, match=r".*domain.*"):
        _setup_integration_domain(3, [[0, 1.0], [1, 2.0]], backend)
    with pytest.raises(ValueError, match=r".*domain.*"):
        _setup_integration_domain(3, custom_domain, "unused")


def test_setup_integration_domain():
    """Test _setup_integration_domain with all possible configurations"""
    _run_tests_with_all_backends(_run_setup_integration_domain_tests)


if __name__ == "__main__":
    try:
        # used to run this test individually
        test_linspace_with_grads()
        test_add_at_indices()
        test_setup_integration_domain()
    except KeyboardInterrupt:
        pass

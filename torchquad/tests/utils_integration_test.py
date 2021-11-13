import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, to_numpy
import importlib
import pytest
import warnings

from integration.utils import _linspace_with_grads, _setup_integration_domain, _RNG
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
        # Tensorflow support for double and numpy support for float aren't
        # implemented yet
        if backend == "numpy":
            supported_precisions = ["double"]
        elif backend == "tensorflow":
            supported_precisions = ["float"]
        else:
            supported_precisions = ["float", "double"]
        if backend == "torch":
            enable_cuda()
        for precision in supported_precisions:
            if backend in ["torch", "jax"]:
                set_precision(precision, backend=backend)
            dtype_name = {"double": "float64", "float": "float32"}[precision]
            # Iterate over arguments in an inner loop here instead of an outer
            # loop so that there are less switches between backends
            for kwargs in func_extra_args:
                func(dtype_name=dtype_name, backend=backend, **kwargs)


def _run_linspace_with_grads_tests(dtype_name, backend, requires_grad):
    """
    Test _linspace_with_grads with the given dtype, numerical backend and
    requires_grad argument
    """
    print(
        f"Testing _linspace_with_grads; backend: {backend}, requires_grad:"
        f" {requires_grad}, precision: {dtype_name}"
    )
    start = anp.array(-2.0, like=backend)
    stop = anp.array(-1.0, like=backend)
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
    custom_domain = anp.array([[0.0, 1.0], [1.0, 2.0]], like=backend)
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


def _run_RNG_tests(dtype_name, backend):
    """
    Test the random number generator with the given dtype and numerical backend
    * With the same seed, the same numbers should be generated
    * With different seeds, different numbers should be generated
    * If seed is None / omitted, the RNG should be randomly seeded
    """
    generateds = [
        _RNG(backend, 547).uniform(size=[3, 9]),
        _RNG(backend, None).uniform(size=[3, 9]),
        _RNG(backend, 547).uniform(size=[3, 9]),
        _RNG(backend).uniform(size=[3, 9]),
        _RNG(backend, 42).uniform(size=[3, 9]),
    ]
    numpy_arrs = list(map(to_numpy, generateds))

    # Validity of the backend, dtype, shape and values range
    assert all(infer_backend(arr) == backend for arr in generateds)
    assert all(get_dtype_name(arr) == dtype_name for arr in generateds)
    assert all(arr.shape == (3, 9) for arr in generateds)
    assert all(0.0 <= x <= 1.0 for arr in numpy_arrs for x in arr.ravel())

    # Test if the seed argument leads to consistent results and
    # if omitting a seed leads to random numbers
    assert anp.array_equal(numpy_arrs[0], numpy_arrs[2])
    for i1 in range(len(generateds)):
        for i2 in range(i1 + 1, len(generateds)):
            if i1 == 0 and i2 == 2:
                continue
            # With a very low probability this may fail
            assert not anp.array_equal(numpy_arrs[i1], numpy_arrs[i2])


def test_RNG():
    """Test the random number generator with all possible configurations"""
    _run_tests_with_all_backends(_run_RNG_tests)


if __name__ == "__main__":
    try:
        # used to run this test individually
        test_linspace_with_grads()
        test_setup_integration_domain()
        test_RNG()
    except KeyboardInterrupt:
        pass

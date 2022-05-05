import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, to_backend_dtype, to_numpy

from integration.rng import RNG

from helper_functions import setup_test_for_backend


def _run_RNG_tests(backend, dtype_name):
    """
    Test the random number generator with the given numerical backend
    * With the same seed, the same numbers should be generated
    * With different seeds, different numbers should be generated
    * If seed is None / omitted, the RNG should be randomly seeded
    """
    backend_dtype = to_backend_dtype(dtype_name, like=backend)
    size = [3, 9]
    generateds = [
        RNG(backend, 547).uniform(size=size, dtype=backend_dtype),
        RNG(backend, None).uniform(size=size, dtype=backend_dtype),
        RNG(backend, 547).uniform(size=size, dtype=backend_dtype),
        RNG(backend).uniform(size=size, dtype=backend_dtype),
        RNG(backend, 42).uniform(size=size, dtype=backend_dtype),
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


test_rng_jax_f32 = setup_test_for_backend(_run_RNG_tests, "jax", "float32")
test_rng_jax_f64 = setup_test_for_backend(_run_RNG_tests, "jax", "float64")
test_rng_numpy_f32 = setup_test_for_backend(_run_RNG_tests, "numpy", "float32")
test_rng_numpy_f64 = setup_test_for_backend(_run_RNG_tests, "numpy", "float64")
test_rng_torch_f32 = setup_test_for_backend(_run_RNG_tests, "torch", "float32")
test_rng_torch_f64 = setup_test_for_backend(_run_RNG_tests, "torch", "float64")
test_rng_tensorflow_f32 = setup_test_for_backend(
    _run_RNG_tests, "tensorflow", "float32"
)
test_rng_tensorflow_f64 = setup_test_for_backend(
    _run_RNG_tests, "tensorflow", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_rng_numpy_f32()
    test_rng_numpy_f64()
    test_rng_torch_f32()
    test_rng_torch_f64()
    test_rng_jax_f32()
    test_rng_jax_f64()
    test_rng_tensorflow_f32()
    test_rng_tensorflow_f64()

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, to_backend_dtype, to_numpy
import pytest

from torchquad.integration.rng import RNG

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


def _run_torch_save_state_tests(backend, dtype_name):
    """
    Test torch_save_state functionality for PyTorch backend
    """
    if backend != "torch":
        pytest.skip("torch_save_state tests only applicable for PyTorch backend")
    
    import torch
    backend_dtype = to_backend_dtype(dtype_name, like=backend)
    size = [2, 3]
    
    # Test that torch_save_state=True works and produces valid arrays
    rng_save_true = RNG(backend, seed=123, torch_save_state=True)
    arr_save_true = rng_save_true.uniform(size=size, dtype=backend_dtype)
    
    # Check basic properties
    assert arr_save_true.shape == (2, 3)
    assert all(0.0 <= x <= 1.0 for x in to_numpy(arr_save_true).ravel())
    
    # Test that torch_save_state=False works and produces valid arrays
    rng_save_false = RNG(backend, seed=123, torch_save_state=False)
    arr_save_false = rng_save_false.uniform(size=size, dtype=backend_dtype)
    
    # Check basic properties
    assert arr_save_false.shape == (2, 3)
    assert all(0.0 <= x <= 1.0 for x in to_numpy(arr_save_false).ravel())
    
    # Test that both modes produce some randomness
    arr2_save_true = rng_save_true.uniform(size=size, dtype=backend_dtype)
    arr2_save_false = rng_save_false.uniform(size=size, dtype=backend_dtype)
    
    # Second calls should be different from first calls
    assert not anp.array_equal(to_numpy(arr_save_true), to_numpy(arr2_save_true))
    assert not anp.array_equal(to_numpy(arr_save_false), to_numpy(arr2_save_false))


def _run_jax_key_tests(backend, dtype_name):
    """
    Test JAX key get/set functionality
    """
    if backend != "jax":
        pytest.skip("JAX key tests only applicable for JAX backend")
    
    from jax.random import PRNGKey
    backend_dtype = to_backend_dtype(dtype_name, like=backend)
    size = [2, 3]
    
    rng = RNG(backend, seed=42)
    
    # Test getting the key
    key1 = rng.jax_get_key()
    assert key1 is not None
    
    # Generate some numbers
    arr1 = rng.uniform(size=size, dtype=backend_dtype)
    
    # Key should have changed after generation
    key2 = rng.jax_get_key()
    assert not anp.array_equal(to_numpy(key1), to_numpy(key2))
    
    # Test setting the key back
    rng.jax_set_key(key1)
    key3 = rng.jax_get_key()
    assert anp.array_equal(to_numpy(key1), to_numpy(key3))
    
    # Should generate the same numbers again
    arr2 = rng.uniform(size=size, dtype=backend_dtype)
    assert anp.array_equal(to_numpy(arr1), to_numpy(arr2))


def _run_edge_case_tests(backend, dtype_name):
    """
    Test edge cases and error conditions
    """
    backend_dtype = to_backend_dtype(dtype_name, like=backend)
    
    # Test with zero size
    rng = RNG(backend, seed=42)
    arr = rng.uniform(size=[0], dtype=backend_dtype)
    assert arr.shape == (0,)
    
    # Test with large size
    large_size = [10, 10]
    arr_large = rng.uniform(size=large_size, dtype=backend_dtype)
    assert arr_large.shape == (10, 10)
    assert all(0.0 <= x <= 1.0 for x in to_numpy(arr_large).ravel())
    
    # Test multiple calls produce different results (unless seeded identically)
    arr1 = rng.uniform(size=[5], dtype=backend_dtype)
    arr2 = rng.uniform(size=[5], dtype=backend_dtype)
    assert not anp.array_equal(to_numpy(arr1), to_numpy(arr2))


def _run_backend_consistency_tests(backend, dtype_name):
    """
    Test that the uniform method generates consistent results with same seed
    """
    backend_dtype = to_backend_dtype(dtype_name, like=backend)
    size = [3, 4]
    
    if backend == "torch":
        # For PyTorch, test torch_save_state functionality for consistency
        rng1 = RNG(backend, seed=12345, torch_save_state=True)
        rng2 = RNG(backend, seed=12345, torch_save_state=True)
        
        # Generate arrays - first calls should be equal
        arr1_first = rng1.uniform(size=size, dtype=backend_dtype)
        arr2_first = rng2.uniform(size=size, dtype=backend_dtype)
        assert anp.array_equal(to_numpy(arr1_first), to_numpy(arr2_first))
        
        # Test that consecutive calls from same RNG are different
        arr1_second = rng1.uniform(size=size, dtype=backend_dtype)
        assert not anp.array_equal(to_numpy(arr1_first), to_numpy(arr1_second))
    else:
        # For other backends, test normal seeding behavior
        rng1 = RNG(backend, seed=12345)
        rng2 = RNG(backend, seed=12345)
        
        # Generate first arrays - should be equal
        arr1 = rng1.uniform(size=size, dtype=backend_dtype)
        arr2 = rng2.uniform(size=size, dtype=backend_dtype)
        assert anp.array_equal(to_numpy(arr1), to_numpy(arr2))
        
        # Test with different seed
        rng3 = RNG(backend, seed=54321)
        arr3 = rng3.uniform(size=size, dtype=backend_dtype)
        assert not anp.array_equal(to_numpy(arr1), to_numpy(arr3))


# Original tests
test_rng_jax_f32 = setup_test_for_backend(_run_RNG_tests, "jax", "float32")
test_rng_jax_f64 = setup_test_for_backend(_run_RNG_tests, "jax", "float64")
test_rng_numpy_f32 = setup_test_for_backend(_run_RNG_tests, "numpy", "float32")
test_rng_numpy_f64 = setup_test_for_backend(_run_RNG_tests, "numpy", "float64")
test_rng_torch_f32 = setup_test_for_backend(_run_RNG_tests, "torch", "float32")
test_rng_torch_f64 = setup_test_for_backend(_run_RNG_tests, "torch", "float64")
test_rng_tensorflow_f32 = setup_test_for_backend(_run_RNG_tests, "tensorflow", "float32")
test_rng_tensorflow_f64 = setup_test_for_backend(_run_RNG_tests, "tensorflow", "float64")

# Additional comprehensive tests
test_torch_save_state_f32 = setup_test_for_backend(_run_torch_save_state_tests, "torch", "float32")
test_torch_save_state_f64 = setup_test_for_backend(_run_torch_save_state_tests, "torch", "float64")

test_jax_key_f32 = setup_test_for_backend(_run_jax_key_tests, "jax", "float32")
test_jax_key_f64 = setup_test_for_backend(_run_jax_key_tests, "jax", "float64")

test_edge_cases_numpy_f32 = setup_test_for_backend(_run_edge_case_tests, "numpy", "float32")
test_edge_cases_numpy_f64 = setup_test_for_backend(_run_edge_case_tests, "numpy", "float64")
test_edge_cases_torch_f32 = setup_test_for_backend(_run_edge_case_tests, "torch", "float32")
test_edge_cases_torch_f64 = setup_test_for_backend(_run_edge_case_tests, "torch", "float64")
test_edge_cases_jax_f32 = setup_test_for_backend(_run_edge_case_tests, "jax", "float32")
test_edge_cases_jax_f64 = setup_test_for_backend(_run_edge_case_tests, "jax", "float64")
test_edge_cases_tensorflow_f32 = setup_test_for_backend(_run_edge_case_tests, "tensorflow", "float32")
test_edge_cases_tensorflow_f64 = setup_test_for_backend(_run_edge_case_tests, "tensorflow", "float64")

test_consistency_numpy_f32 = setup_test_for_backend(_run_backend_consistency_tests, "numpy", "float32")
test_consistency_numpy_f64 = setup_test_for_backend(_run_backend_consistency_tests, "numpy", "float64")
test_consistency_torch_f32 = setup_test_for_backend(_run_backend_consistency_tests, "torch", "float32")
test_consistency_torch_f64 = setup_test_for_backend(_run_backend_consistency_tests, "torch", "float64")
test_consistency_jax_f32 = setup_test_for_backend(_run_backend_consistency_tests, "jax", "float32")
test_consistency_jax_f64 = setup_test_for_backend(_run_backend_consistency_tests, "jax", "float64")
test_consistency_tensorflow_f32 = setup_test_for_backend(_run_backend_consistency_tests, "tensorflow", "float32")
test_consistency_tensorflow_f64 = setup_test_for_backend(_run_backend_consistency_tests, "tensorflow", "float64")


if __name__ == "__main__":
    # used to run this test individually
    # Original tests
    test_rng_numpy_f32()
    test_rng_numpy_f64()
    test_rng_torch_f32()
    test_rng_torch_f64()
    test_rng_jax_f32()
    test_rng_jax_f64()
    test_rng_tensorflow_f32()
    test_rng_tensorflow_f64()
    
    # Additional comprehensive tests
    test_torch_save_state_f32()
    test_torch_save_state_f64()
    test_jax_key_f32()
    test_jax_key_f64()
    
    # Edge case tests
    test_edge_cases_numpy_f32()
    test_edge_cases_torch_f32()
    test_edge_cases_jax_f32()
    test_edge_cases_tensorflow_f32()
    
    # Consistency tests
    test_consistency_numpy_f32()
    test_consistency_torch_f32()
    test_consistency_jax_f32()
    test_consistency_tensorflow_f32()

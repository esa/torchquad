import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.rng import RNG
from integration.vegas_stratification import VEGASStratification

from helper_functions import setup_test_for_backend


def _run_vegas_stratification_checks(backend, dtype_name):
    """Test if the VEGASStratification methods work correctly"""
    print(f"Testing VEGASStratification with {backend}, {dtype_name}")
    dtype_float = to_backend_dtype(dtype_name, like=backend)
    dtype_int = to_backend_dtype("int64", like=backend)
    dim = 3
    strat = VEGASStratification(
        1000,
        dim=dim,
        rng=RNG(backend=backend, seed=0),
        backend=backend,
        dtype=dtype_float,
    )

    # Test if get_NH works correctly for a fresh VEGASStratification
    neval = strat.get_NH(4000)
    assert neval.dtype == dtype_int
    assert neval.shape == (strat.N_cubes,)
    assert (
        anp.max(anp.abs(neval - neval[0])) == 0
    ), "Varying number of evaluations per hypercube for a fresh VEGASStratification"

    # Test if sample point calculation works correctly for a
    # fresh VEGASStratification
    y = strat.get_Y(neval)
    assert y.dtype == dtype_float
    assert y.shape == (anp.sum(neval), dim)
    assert anp.all(y >= 0.0) and anp.all(y <= 1.0), "Sample points are out of bounds"

    # Test accumulate_weight
    # Use exp to get a peak in a corner
    f_eval = anp.prod(anp.exp(y), axis=1)
    jf, jf2 = strat.accumulate_weight(neval, f_eval)
    assert jf.dtype == jf2.dtype == dtype_float
    assert jf.shape == jf2.shape == (strat.N_cubes,)
    assert anp.min(jf2) >= 0.0, "Sums of squared values should be non-negative"
    assert (
        anp.min(jf**2 - jf2) >= 0.0
    ), "Squared sums should be bigger than summed squares"

    # Test the dampened sample counts update
    strat.update_DH()
    assert strat.dh.shape == (strat.N_cubes,)
    assert strat.dh.dtype == dtype_float
    assert anp.min(strat.dh) >= 0.0, "Invalid probabilities for hypercubes"
    assert anp.abs(strat.dh.sum() - 1.0) < 4e-7, "Invalid probabilities for hypercubes"
    assert (
        strat.dh[-1] > strat.dh[0]
    ), "The hypercube at the peak should have a higher probability to get points"

    # Test if get_NH still works correctly
    neval = strat.get_NH(4000)
    assert neval.dtype == dtype_int
    assert neval.shape == (strat.N_cubes,)
    assert neval[-1] > neval[0], "The hypercube at the peak should have more points"

    # Test if sample point calculation still works correctly
    y = strat.get_Y(neval)
    assert y.dtype == dtype_float
    assert y.shape == (anp.sum(neval), dim)
    assert anp.all(y >= 0.0) and anp.all(y <= 1.0), "Sample points are out of bounds"


test_vegas_stratification_numpy_f32 = setup_test_for_backend(
    _run_vegas_stratification_checks, "numpy", "float32"
)
test_vegas_stratification_numpy_f64 = setup_test_for_backend(
    _run_vegas_stratification_checks, "numpy", "float64"
)
test_vegas_stratification_torch_f32 = setup_test_for_backend(
    _run_vegas_stratification_checks, "torch", "float32"
)
test_vegas_stratification_torch_f64 = setup_test_for_backend(
    _run_vegas_stratification_checks, "torch", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_vegas_stratification_numpy_f32()
    test_vegas_stratification_numpy_f64()
    test_vegas_stratification_torch_f32()
    test_vegas_stratification_torch_f64()

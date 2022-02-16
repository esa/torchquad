import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.vegas_map import VEGASMap

from helper_functions import setup_test_for_backend


def _check_tensor_similarity(a, b, err_abs_max=0.0, expected_dtype=None):
    """Check if two tensors have the same dtype, shape and are equal up to a specified error"""
    if expected_dtype:
        assert a.dtype == b.dtype == expected_dtype
    else:
        assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert anp.max(anp.abs(a - b)) <= err_abs_max


def _run_vegas_map_checks(backend, dtype_name):
    """Test if the VEGASMap methods work correctly while running a map update for example integrand output"""
    print(f"Testing VEGASMap with {backend}, {dtype_name}")
    dtype_float = to_backend_dtype(dtype_name, like=backend)
    dtype_int = to_backend_dtype("int64", like=backend)
    dim = 3
    N_intervals = 20
    vegasmap = VEGASMap(N_intervals, dim, backend, dtype_float)

    y = anp.array(
        [[0.8121, 0.4319, 0.1612], [0.4746, 0.6501, 0.9241], [0.6143, 0.0724, 0.5818]],
        dtype=dtype_float,
        like=backend,
    )

    # Test _get_interval_ID and _get_interval_offset for the fresh VEGAS map
    ID_expected = anp.array(
        [[16, 8, 3], [9, 13, 18], [12, 1, 11]],
        dtype=dtype_int,
        like=backend,
    )
    off_expected = anp.array(
        [[0.2420, 0.6380, 0.2240], [0.4920, 0.0020, 0.4820], [0.2860, 0.4480, 0.6360]],
        dtype=dtype_float,
        like=backend,
    )
    ID = vegasmap._get_interval_ID(y)
    _check_tensor_similarity(ID, ID_expected, 0, dtype_int)
    off = vegasmap._get_interval_offset(y)
    _check_tensor_similarity(off, off_expected, 6e-5, dtype_float)

    # Test get_X for the fresh VEGAS map
    # Initially it should not change the points
    _check_tensor_similarity(vegasmap.get_X(y), y, 3e-7, dtype_float)

    # Get example point and function values
    N_per_dim = 100
    y = anp.linspace(0.0, 0.99999, N_per_dim, dtype=dtype_float, like=backend)
    y = anp.meshgrid(*([y] * dim))
    y = anp.stack([mg.ravel() for mg in y], axis=1, like=backend)
    # Use exp to get a peak in a corner
    f_eval = anp.prod(anp.exp(y), axis=1)

    # Test get_Jac for a fresh VEGAS map
    jac = vegasmap.get_Jac(y)
    assert jac.shape == (N_per_dim**dim,)
    assert jac.dtype == dtype_float
    assert anp.max(anp.abs(jac - 1.0)) < 1e-14

    # Test vegasmap.accumulate_weight for a fresh VEGAS map
    jf_vec = f_eval * jac
    jf_vec2 = jf_vec**2
    vegasmap.accumulate_weight(y, jf_vec2)
    assert vegasmap.weights.dtype == dtype_float
    assert vegasmap.weights.shape == (dim, N_intervals)
    # The weights should be monotonically increasing for the given f_evals and
    # vegasmap
    assert anp.min(vegasmap.weights[:, 1:] - vegasmap.weights[:, :-1]) > 0.0
    assert vegasmap.counts.dtype == dtype_int
    assert vegasmap.counts.shape == (dim, N_intervals)
    # The counts are all 50000 here since y are grid points and the VEGAS map
    # does not yet warp points
    assert anp.max(anp.abs(vegasmap.counts - 50000)) == 0

    # Test vegasmap._smooth_map
    weights = anp.array(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=dtype_float,
        like=backend,
    )
    counts = anp.ones(weights.shape, dtype=dtype_int, like=backend)
    alpha = 0.5
    smoothed_weights_expected = anp.array(
        [
            [0.0, 0.0, 0.54913316, 0.75820765, 0.77899047, 0.77899047],
            [-0.0, -0.0, 0.64868024, 0.93220967, 0.64868024, -0.0],
        ],
        dtype=dtype_float,
        like=backend,
    )
    smoothed_weights = VEGASMap._smooth_map(weights, counts, alpha)
    _check_tensor_similarity(
        smoothed_weights, smoothed_weights_expected, 3e-7, dtype_float
    )

    # Test if vegasmap.update_map changes the edge locations and distances
    # correctly
    vegasmap.update_map()
    # The outermost edge locations must match the domain [0,1]^dim
    unit_domain = anp.array([[0.0, 1.0]] * dim, dtype=dtype_float, like=backend)
    _check_tensor_similarity(
        vegasmap.x_edges[:, [0, -1]], unit_domain, 0.0, dtype_float
    )
    assert vegasmap.x_edges.shape == (dim, N_intervals + 1), "Invalid number of edges"
    assert vegasmap.dx_edges.shape == (
        dim,
        N_intervals,
    ), "Invalid number of edge distances"
    assert vegasmap.dx_edges.dtype == dtype_float
    assert (
        anp.max(anp.abs(anp.sum(vegasmap.dx_edges, axis=1) - 1.0)) < 3e-7
    ), "In each dimension the edge distances should sum up to one."
    assert anp.min(vegasmap.dx_edges) > 0.0, "Non-positive edge distance"
    # The absolute value of the given integrand is monotonically increasing in
    # each dimension, so calculated interval sizes should monotonically decrease
    assert (
        anp.max(vegasmap.dx_edges[:, 1:] - vegasmap.dx_edges[:, :-1]) < 0.0
    ), "Edge distances should shrink towards the peak"

    # Test if the new mapping of points works correctly
    x = vegasmap.get_X(y)
    assert x.dtype == dtype_float
    assert x.shape == y.shape
    assert anp.max(anp.abs(x[0])) == 0.0, "Boundary point was remapped"


test_vegas_map_numpy_f32 = setup_test_for_backend(
    _run_vegas_map_checks, "numpy", "float32"
)
test_vegas_map_numpy_f64 = setup_test_for_backend(
    _run_vegas_map_checks, "numpy", "float64"
)
test_vegas_map_torch_f32 = setup_test_for_backend(
    _run_vegas_map_checks, "torch", "float32"
)
test_vegas_map_torch_f64 = setup_test_for_backend(
    _run_vegas_map_checks, "torch", "float64"
)


if __name__ == "__main__":
    # used to run this test individually
    test_vegas_map_numpy_f32()
    test_vegas_map_numpy_f64()
    test_vegas_map_torch_f32()
    test_vegas_map_torch_f64()

import warnings

from autoray import numpy as anp


import numpy as np
from pathlib import Path
from .rng import RNG

from functools import lru_cache

_VALID_BACKENDS = ("torch", "numpy", "jax", "tensorflow")


class Sobol:
    """A scrambled Sobol low-discrepancy sampler, shaped like :class:`RNG`.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into quasi-Monte Carlo (QMC): Sobol points cover the
    unit hypercube far more evenly than pseudo-random draws, so for smooth
    integrands the error shrinks close to ``O(1/N)`` instead of the ``O(1/sqrt(N))``
    of plain Monte Carlo.

    Like :class:`RNG`, an instance exposes ``uniform(size, dtype)`` returning
    points in ``[0, 1)`` as a backend tensor, so it is a drop-in replacement for
    the sampler ``MonteCarlo`` uses internally.

    Points are generated with PyTorch's ``torch.quasirandom.SobolEngine`` for the
    ``torch`` backend and with ``scipy.stats.qmc.Sobol`` for the others (converted
    to the requested backend). As with plain Monte Carlo the sample points are
    constants, so gradients still flow through the integrand and the integration
    domain; only the point *placement* differs.

    Notes:
        - The number of points should be a power of two for the Sobol balance
          properties to hold; SciPy emits a warning otherwise.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path, not
          the JIT-compiled one (which builds its own RNG internally).
        - Per-backend Sobol implementations use different scrambling, so results
          are reproducible for a fixed ``seed`` within a backend but do not match
          bit-for-bit across backends.
    """

    def __init__(self, backend, seed=None, scramble=True):
        """Initialize a Sobol sampler.

        Args:
            backend (string): Numerical backend, e.g. "torch". Must match the
                backend of the integration domain it will be used with.
            seed (int or None, optional): Seed for the scrambling. If None, the
                scrambling is randomised. Defaults to None.
            scramble (bool, optional): Whether to apply Owen scrambling, which
                randomises the sequence while preserving its low discrepancy and
                yields an unbiased estimator. Defaults to True.
        """
        self._backend = backend
        self._seed = seed
        self._scramble = scramble

    def uniform(self, size, dtype):
        """Draw Sobol points in ``[0, 1)``.

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (backend dtype): Floating point dtype of the returned tensor.

        Returns:
            backend tensor: ``[number_of_points, dim]`` Sobol points in ``[0, 1)``.
        """
        number_of_points, dim = int(size[0]), int(size[1])

        # Warn on non-power-of-two counts uniformly across backends: SciPy already
        # warns on its own path, but torch's SobolEngine would silently return a
        # sequence prefix whose balance properties do not hold.
        if self._backend == "torch" and number_of_points & (number_of_points - 1) != 0:
            warnings.warn(
                "The balance properties of Sobol points require the number of "
                f"points to be a power of 2, but {number_of_points} was requested.",
                stacklevel=2,
            )

        if self._backend == "torch":
            # SobolEngine keeps the points as native torch tensors (mirrors how
            # RNG special-cases torch); no NumPy round-trip on the GPU.
            import torch

            engine = torch.quasirandom.SobolEngine(
                dimension=dim, scramble=self._scramble, seed=self._seed
            )
            points = engine.draw(number_of_points, dtype=dtype)
            # SobolEngine always draws on the CPU; move the points onto the
            # current default device (e.g. CUDA) so they line up with the
            # integration domain, matching what torch.rand does in RNG.
            return points.to(torch.empty(0).device)

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the constant points onto the requested backend.
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=self._scramble, seed=self._seed)
        points = sampler.random(number_of_points)
        return anp.array(points, dtype=dtype, like=self._backend)


DATA_DIR = Path(__file__).resolve().parent


_MIN_POINTS = 1024
_MAX_POINTS = 1048576
_LATTICE_FILE = "lattice-33002-1024-1048576.npz"


@lru_cache(maxsize=None)
def _load_full_generating_vector(filename):
    """Load and cache the full generating vector array from disk (once)."""
    path = DATA_DIR / filename
    with np.load(path) as data:
        generating_vector = data["generating_vector"]
    return generating_vector.astype(np.uint64)


def _max_dimension(filename=_LATTICE_FILE):
    """Return the largest dimension supported by the generating vector file."""
    return _load_full_generating_vector(filename).shape[0]


def load_lattice_vector(d, filename=_LATTICE_FILE):
    """Load the first d components of the generating vector z from a
    generating vectors file for rank-1 lattice rules.

    The full array is loaded from disk only once (cached); each call
    afterwards just slices the already-loaded array. The valid range for
    ``d`` is derived from the file itself, not hard-coded. The file must
    contain a single array ``generating_vector`` with one component per
    dimension.

    Args:
        d (int): Desired dimension.
        filename (str, optional): Path to the generating vectors ``.npz``
            file. Defaults to ``_LATTICE_FILE``.

    Returns:
        np.ndarray: Array [z1, z2, ..., zd] with dtype uint64.

    Raises:
        ValueError: If d is not strictly positive, or if d exceeds the
            number of components available in the file.
    """
    if d <= 0:
        raise ValueError("The dimension d must be strictly positive.")

    generating_vector = _load_full_generating_vector(filename)

    max_d = generating_vector.shape[0]

    if d > max_d:
        raise ValueError(
            f"Requested dimension d={d}, but the file contains only {max_d} components."
        )

    return generating_vector[:d]


class Lattice:
    """A randomly shifted rank-1 lattice low-discrepancy sampler.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into quasi-Monte Carlo (QMC). Rank-1 lattice points
    cover the unit hypercube more evenly than pseudo-random draws, which can
    reduce the integration error substantially for sufficiently smooth
    integrands.

    Like :class:`RNG`, an instance exposes ``uniform(size, dtype)`` returning
    points in ``[0, 1)`` as a backend tensor, so it is a drop-in replacement for
    the sampler ``MonteCarlo`` uses internally. The construction is a few lines
    of array arithmetic, dispatched across backends via ``autoray``, so it runs
    natively on all four supported backends without any per-backend branching
    or additional dependency.

    The generating vector is loaded from the file
    ``lattice-33002-1024-1048576.npz``, obtained from the *Lattice Rule
    Generating Vectors* collection maintained by Frances Y. Kuo. This file
    provides an extensible rank-1 lattice rule for dimensions up to 9125 and
    sample sizes from 1024 to 1048576.

    The generating vector was constructed using a component-by-component (CBC)
    algorithm. The construction minimizes, as much as possible, the
    shift-averaged worst-case integration error in a weighted unanchored Sobolev
    space. The file uses order-3 weights with
    ``Gamma_1 = Gamma_2 = 1`` and ``Gamma_3 = 0.5``.

    For a generating vector ``z`` and ``N`` points, the ``i``-th lattice point
    is computed componentwise as

    ``x_i = frac(i * z / N)``,

    where ``i = 0, ..., N - 1`` and ``frac`` denotes the fractional part. Since
    this is an extensible lattice rule, this simple construction is valid when
    the total number of points is an exact power of two.

    When ``shift=True``, a random shift is sampled uniformly from the unit
    hypercube on every call to ``uniform`` (added modulo one to every lattice
    point). This preserves the lattice structure while randomizing the point
    set, making the resulting randomized QMC estimator unbiased for each call.
    With a fixed ``seed``, the shift drawn on each call is identical, so
    successive calls reproduce the same points; without a seed, each call
    draws its own independent shift.

    When ``tent=True``, a Baker's transform (tent map) is applied to every
    point *after* the shift: ``phi(x) = 1 - |2x - 1|``. Lattice rules recover
    their best convergence rate for *periodic* integrands; for a non-periodic
    integrand, this transform periodizes it (without changing the value of
    the integral), which can dramatically reduce the integration error for
    smooth non-periodic integrands. Defaults to False, since it changes which
    points are evaluated and is not always beneficial (e.g. for an integrand
    that is already periodic on the domain).

    **Range with** ``tent=True``: the tent map's peak value is exactly 1,
    reached whenever a point lands exactly on 0.5. With ``shift=False``, this
    happens systematically (not by rare chance): the bundled generating
    vector's first component is 1, so for any valid (power-of-two) number of
    points ``N``, index ``i = N/2`` lands exactly on 0.5 in the first
    dimension. With ``shift=True`` this would additionally require the
    continuous random shift to land a point exactly on 0.5, which has
    probability zero but is not structurally excluded. Tented points
    therefore lie in the **closed** interval ``[0, 1]``, unlike every other
    configuration of this sampler (``tent=False``, with or without
    ``shift``), which is provably confined to the half-open ``[0, 1)``: the
    unshifted construction is bounded above by ``(N-1)/N < 1`` by integer
    arithmetic, and the shifted construction's final ``% 1.0`` maps into
    ``[0, 1)`` by definition of the modulo operation, regardless of its
    input.

    As with plain Monte Carlo, the sample points are constants, so gradients
    still flow through the integrand and the integration domain; only the point
    placement differs.

    Notes:
        - The dimension must be a whole number, and no larger than what the
          loaded generating vector file supports (currently 9125; see
          :func:`load_lattice_vector`, which derives this bound from the file
          itself rather than a hard-coded constant).
        - The number of points must be a whole number between
          :data:`_MIN_POINTS` and :data:`_MAX_POINTS`, inclusive.
        - The number of points must be a power of two for the simple
          construction above to be valid for this extensible lattice rule.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path, not
          the JIT-compiled one (which builds its own RNG internally).
        - The random shift is drawn via a fresh, per-call :class:`RNG`
          instance rather than any backend's global RNG state, so
          constructing or calling this sampler never affects unrelated
          random draws elsewhere in your program.
        - Unshifted points are bit-identical across backends; shifted points
          are reproducible for a fixed ``seed`` within a backend, but the
          shift itself is drawn independently per backend, so shifted points
          do not match bit-for-bit across backends.
    """

    def __init__(self, backend, seed=None, shift=True, tent=False):
        """Initialize a rank-1 lattice sampler.

        Args:
            backend (string): Numerical backend. Must be one of "torch",
                "numpy", "jax", or "tensorflow", and match the backend of the
                integration domain it will be used with.
            seed (int or None, optional): Seed used to generate the random
                shift, redrawn on every call to ``uniform``. A fixed seed
                makes that draw reproducible across calls; if None, each
                call gets an independent random shift. Defaults to None.
            shift (bool, optional): Whether to apply a random shift modulo one
                to the lattice points. Defaults to True.
            tent (bool, optional): Whether to apply a Baker's transform (tent
                map) to the points after the shift, which periodizes a
                non-periodic integrand and can substantially improve
                convergence for smooth non-periodic integrands. Unlike every
                other configuration of this sampler, points returned with
                ``tent=True`` lie in the closed interval [0, 1] rather than
                [0, 1): the tent map's peak is exactly 1, reached whenever a
                point lands exactly on 0.5 .
                Defaults to False.

        Raises:
            ValueError: If backend is not one of "torch", "numpy", "jax", or
                "tensorflow".
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f'Lattice only supports backends {_VALID_BACKENDS}, but got "{backend}".'
            )
        self._backend = backend
        self._seed = seed
        self._shift = shift
        self._tent = tent

    def uniform(self, size, dtype):
        """Draw rank-1 lattice points in ``[0, 1)`` (or ``[0, 1]`` if
        ``tent=True``).

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (backend dtype): Floating-point dtype of the returned tensor.

        Returns:
            backend tensor: ``[number_of_points, dim]`` lattice points in
            ``[0, 1)``, or in the closed interval ``[0, 1]`` if
            ``tent=True``.

        Raises:
            ValueError: If the number of points is not between
                :data:`_MIN_POINTS` and :data:`_MAX_POINTS`, is not a power of
                2, or if the dimension is invalid for the loaded generating
                vector file (see :func:`load_lattice_vector`).
        """
        number_of_points, dim = int(size[0]), int(size[1])

        if not _MIN_POINTS <= number_of_points <= _MAX_POINTS:
            raise ValueError(
                f"The number of points must be between {_MIN_POINTS} and {_MAX_POINTS}, "
                f"but got {number_of_points}."
            )

        if number_of_points & (number_of_points - 1) != 0:
            raise ValueError(
                "The simple lattice construction used here is only valid when the "
                f"number of points is a power of 2, but {number_of_points} was "
                "requested."
            )

        z_np = load_lattice_vector(d=dim, filename=_LATTICE_FILE)

        z = anp.astype(anp.array(z_np, like=self._backend), "int64")
        index = anp.astype(anp.arange(number_of_points, like=self._backend), "int64")
        mod = (index[:, None] * z[None, :]) % number_of_points
        points = anp.astype(mod, dtype) / number_of_points

        if self._shift:
            shift_values = RNG(backend=self._backend, seed=self._seed).uniform([dim], dtype)
            points = (points + shift_values[None, :]) % 1.0

        if self._tent:
            # Baker's transform: periodizes the integrand so the lattice rule
            # recovers its convergence rate on non-periodic integrands.
            # The peak of this map is exactly 1 (reached when a point lands
            # exactly on 0.5), so the output range becomes the closed [0, 1]
            # here, unlike the [0, 1) guaranteed everywhere else in this
            # method
            points = 1.0 - anp.abs(2.0 * points - 1.0)

        return points

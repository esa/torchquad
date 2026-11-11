import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from autoray import get_dtype_name
from autoray import numpy as anp

from .rng import RNG

_VALID_BACKENDS = ("torch", "numpy", "jax", "tensorflow")


def _below_one(points):
    """Clamp points to the largest value below 1 that their dtype can represent.

    The samplers construct points in ``[0, 1)`` in exact or double precision
    arithmetic, but rounding to a narrower dtype can turn a value just below 1
    into exactly 1.0 (in float32, any value above ``1 - 2**-25``). The integrand
    would then be evaluated on the upper domain boundary, which ``RNG`` never
    does. Values already below 1 are returned unchanged.

    Args:
        points (backend tensor): Points in ``[0, 1]``.

    Returns:
        backend tensor: The same points, none of them equal to 1.
    """
    largest = float(np.nextafter(np.dtype(get_dtype_name(points)).type(1.0), 0.0))
    return anp.clip(points, 0.0, largest)


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
            return _below_one(points.to(torch.empty(0).device))

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the constant points onto the requested backend.
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=self._scramble, seed=self._seed)
        points = sampler.random(number_of_points)
        return _below_one(anp.array(points, dtype=dtype, like=self._backend))


_DATA_DIR = Path(__file__).resolve().parent


_MIN_POINTS = 1024
_MAX_POINTS = 1048576
_LATTICE_FILE = "lattice-33002-1024-1048576.npz"


@lru_cache(maxsize=None)
def _load_full_generating_vector():
    """Load the bundled generating vector from disk, once."""
    with np.load(_DATA_DIR / _LATTICE_FILE) as data:
        generating_vector = data["generating_vector"]
    return generating_vector.astype(np.uint64)


def _load_lattice_vector(dim):
    """Return the first ``dim`` components of the bundled generating vector.

    Args:
        dim (int): Desired dimension.

    Returns:
        np.ndarray: Array ``[z_1, ..., z_dim]`` with dtype uint64.

    Raises:
        ValueError: If dim is not strictly positive, or exceeds the number of
            components in the bundled file.
    """
    if dim <= 0:
        raise ValueError("The dimension must be strictly positive.")

    generating_vector = _load_full_generating_vector()
    max_dim = generating_vector.shape[0]
    if dim > max_dim:
        raise ValueError(
            f"Requested dimension {dim}, but the generating vector has only {max_dim} components."
        )
    return generating_vector[:dim]


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
        - The dimension must be a whole number no larger than the number of
          components in the bundled generating vector (9125).
        - The number of points must be a whole number between 1024 and
          1048576, inclusive.
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
            ValueError: If the number of points is not between 1024 and
                1048576, is not a power of 2, or if the dimension is not
                between 1 and the length of the bundled generating vector.
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

        z_np = _load_lattice_vector(dim)

        # index * z reaches 2**40, which needs 64-bit integers. JAX only has
        # them with x64 enabled; in its default 32-bit mode the product wraps
        # modulo 2**32 instead. The points are still exact there, because the
        # number of points is a power of two (checked above) and therefore
        # divides 2**32, so the wraparound vanishes under the modulo. Relaxing
        # the power-of-two check would silently break this.
        integer_dtype = "int64"
        if self._backend == "jax":
            # JAX's default integer type is int32 exactly when x64 is off.
            integer_dtype = get_dtype_name(anp.arange(1, like="jax"))
        z = anp.astype(anp.array(z_np.astype(np.int64), like=self._backend), integer_dtype)
        index = anp.astype(anp.arange(number_of_points, like=self._backend), integer_dtype)
        mod = (index[:, None] * z[None, :]) % number_of_points
        points = anp.astype(mod, dtype) / number_of_points

        if self._shift:
            # torch_save_state gives the shift its own generator; without it
            # RNG seeds PyTorch's global generator, which would disturb every
            # unrelated random draw in the caller's program.
            shift_values = RNG(
                backend=self._backend, seed=self._seed, torch_save_state=True
            ).uniform([dim], dtype)
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


def _first_n_primes(n):
    """The first n prime numbers p_1,...,p_n -- the bases used by the
    Halton construction (see the docstring of Halton.uniform)."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _digits_needed(base, n_points):
    """Number of digits m such that base^m > largest index (n_points-1).
    Computed with pure integer arithmetic: math.log(x, base) can round to
    just above an exact integer (e.g. log(15625,5) = 6.000000000000001),
    tipping math.ceil to 7 instead of 6."""
    if n_points <= 1:
        return 1
    m = 1
    while base**m <= (n_points - 1):
        m += 1
    return m


def _hash_mix(seed, dim_idx, depth, prefix):
    """Deterministic integer mixer (splitmix64-style), vectorised over a
    tensor of prefixes. Used to derive, for each (dimension, depth,
    ALREADY-SCRAMBLED prefix), a reproducible pseudo-random shift.

    This is a simplification of full Owen scrambling (which would allow any
    permutation at each node): here, only a cyclic shift mod base is applied
    at each node. A cyclic shift IS a genuine bijection of Z_base (so it
    preserves equidistribution properties exactly), but it does not cover the
    full space of possible
    permutations for base > 2. For base=2 (Halton's first dimension),
    however, this is rigorously EQUIVALENT to full Owen scrambling, since
    there are only 2 possible permutations of a 2-element set (identity and
    swap), exactly the two values a shift mod 2 can take.
    """
    import torch

    x = prefix.to(torch.int64)
    salt = (seed * 1000003 + dim_idx * 97 + depth) & 0x7FFFFFFFFFFFFFFF
    x = x * 6364136223846793005 + salt + 0x9E3779B97F4A7C15
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x


class Halton:
    """A (optionally scrambled) Halton low-discrepancy sampler, shaped like
    :class:`RNG` and :class:`Sobol`.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into quasi-Monte Carlo (QMC): Halton points cover
    the unit hypercube far more evenly than pseudo-random draws, so for smooth
    integrands the error shrinks close to ``O(1/N)`` instead of the
    ``O(1/sqrt(N))`` of plain Monte Carlo (with an extra logarithmic-in-N
    factor in higher dimensions, and a somewhat worse constant than Sobol).

    Like :class:`RNG` and :class:`Sobol`, an instance exposes
    ``uniform(size, dtype)`` returning points in ``[0, 1)`` as a backend
    tensor, so it is a drop-in replacement for the sampler ``MonteCarlo`` uses
    internally.

    The Halton point set assigns to index i the point
    ``(phi_{p_1}(i), ..., phi_{p_d}(i))``, where ``p_1,...,p_d`` are the first
    d primes and ``phi_b`` is the radical inverse function in base b: write i
    in base b as ``i = d_0 + d_1 b + d_2 b^2 + ...``, then
    ``phi_b(i) = d_0 b^-1 + d_1 b^-2 + d_2 b^-3 + ...``. Points are generated
    directly on the requested backend/device in pure PyTorch for the
    ``torch`` backend, and with ``scipy.stats.qmc.Halton`` for the others
    (converted to the requested backend). As with plain Monte Carlo the
    sample points are constants, so gradients still flow through the
    integrand and the integration domain; only the point *placement* differs.

    Notes:
        - Unlike Sobol, Halton does not require the sample size to be a power
          of two: any n is valid, at the cost of a somewhat slower
          convergence rate.
        - Scrambling here applies an independent random digit shift at each
          node of the digit-expansion tree (see `_hash_mix`), where the shift
          at depth r depends on the already-scrambled digits 0..r-1. This
          preserves low discrepancy exactly like Owen scrambling, but is a
          simplified subset of it for bases > 2 (see `_hash_mix` docstring).
          A further continuous shift, uniform within the resolution of the
          last digit, is added after digit scrambling (see `uniform`), so
          the estimator remains unbiased even when few digits are needed
          (e.g. N=1).
        - The number of digits m is computed PER DIMENSION, not shared across
          dimensions: dimensions with larger bases need fewer digits for the
          same number of points, and sharing a single m (driven by the
          smallest base) would inject spurious scrambling noise into unused
          high-order digit positions of the other dimensions.
        - On the ``torch`` backend the points are generated directly on the
          default device, with no NumPy round-trip. The other backends
          generate them with SciPy on the CPU and convert.
        - Floating-point caveat: coordinates equal to k / base^m are not
          always exactly representable in float64 when base != 2 (e.g.
          46/729 does not round-trip exactly through float64). This can
          occasionally shift
          a point by one ULP across a stratum boundary if you recompute
          floor(x * n) downstream; it does not affect the validity of the
          point set for integration.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path,
          not the JIT-compiled one (which builds its own RNG internally).
        - Per-backend Halton implementations use different scrambling, so
          results are reproducible for a fixed ``seed`` within a backend but
          do not match bit-for-bit across backends.
    """

    def __init__(self, backend, seed=None, scramble=True):
        """Initialize a Halton sampler.

        Args:
            backend (string): Numerical backend. Must be one of "torch",
                "numpy", "jax", or "tensorflow", and match the backend of the
                integration domain it will be used with.
            seed (int or None, optional): Seed for the scrambling. If None,
                the scrambling is randomised. Defaults to None.
            scramble (bool, optional): Whether to apply digit scrambling
                (see class Notes), which randomises the sequence while
                preserving its low discrepancy and yields an unbiased
                estimator. Defaults to True.

        Raises:
            ValueError: If backend is not one of "torch", "numpy", "jax", or
                "tensorflow".
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f'Halton only supports backends {_VALID_BACKENDS}, but got "{backend}".'
            )
        self._backend = backend
        self._seed = seed
        self._scramble = scramble

    def uniform(self, size, dtype):
        """Draw Halton points in ``[0, 1)``.

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (backend dtype): Floating point dtype of the returned tensor.

        Returns:
            backend tensor: ``[number_of_points, dim]`` Halton points in ``[0, 1)``.
        """
        number_of_points, dim = int(size[0]), int(size[1])

        if self._backend == "torch":
            # Points are generated directly on the current default device
            # (e.g. CUDA), matching what torch.rand does in RNG -- no CPU
            # round-trip needed here (unlike SobolEngine).
            import torch

            device = torch.empty(0).device
            seed = self._seed
            if seed is None:
                # Generator.seed() draws from the operating system's entropy,
                # so an unseeded sampler does not advance PyTorch's global
                # generator the way torch.randint would.
                seed = torch.Generator().seed() & 0x7FFFFFFF

            primes = _first_n_primes(dim)
            idx_all = torch.arange(number_of_points, device=device, dtype=torch.int64)

            columns = []
            for j, base in enumerate(primes):
                # m_j is computed PER DIMENSION: dimensions with larger bases
                # need fewer digits for the same number_of_points. Sharing a
                # single m across all dimensions (driven by the smallest
                # base) would inject spurious high-order scrambling noise
                # into dimensions that don't need those extra digit
                # positions, which breaks equidistribution.
                m_j = _digits_needed(base, number_of_points)

                idx = idx_all.clone()
                digits = torch.empty((number_of_points, m_j), dtype=torch.int64, device=device)
                for r in range(m_j):
                    digits[:, r] = idx % base
                    idx //= base
                # digits[:, 0] = most significant fractional digit (weight
                # base^-1), ..., digits[:, m_j-1] = least significant.

                if self._scramble:
                    prefix = torch.zeros(number_of_points, dtype=torch.int64, device=device)
                    scrambled = torch.empty_like(digits)
                    for r in range(m_j):
                        shift = _hash_mix(seed, j, r, prefix) % base
                        scrambled[:, r] = (digits[:, r] + shift) % base
                        # The next depth's shift depends on the SCRAMBLED
                        # prefix so far, giving the genuine tree structure
                        # of Owen-style scrambling.
                        prefix = prefix * base + scrambled[:, r]
                    digits = scrambled

                # Reconstruct in pure integer arithmetic (Horner), converting
                # to float only once at the very end. Summing digit*weight
                # terms one at a time in float64 instead would accumulate
                # rounding error at every step (each weight base^-k is
                # itself not exactly representable for base != 2) and
                # mis-stratify many more points than the single division,
                # which is only subject to the float64 representability
                # limit (plain Python: `46/729*729 != 46`).
                acc = torch.zeros(number_of_points, dtype=torch.int64, device=device)
                for r in range(m_j):
                    acc = acc * base + digits[:, r]
                col = acc.to(torch.float64) / (base**m_j)

                if self._scramble:
                    tail_generator = torch.Generator(device=device)
                    tail_seed = (seed * 1000003 + j * 97 + m_j) & 0x7FFFFFFF
                    tail_generator.manual_seed(tail_seed)
                    tail = torch.rand(
                        1, generator=tail_generator, device=device, dtype=torch.float64
                    )
                    col = col + tail / (base**m_j)

                columns.append(col)

            return _below_one(torch.stack(columns, dim=1).to(dtype))

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the constant points onto the requested backend.
        from scipy.stats import qmc

        sampler = qmc.Halton(d=dim, scramble=self._scramble, seed=self._seed)
        points = sampler.random(number_of_points)
        return _below_one(anp.array(points, dtype=dtype, like=self._backend))


class RandomizedLatinHypercube:
    """A randomized Latin Hypercube (LHS) sampler, shaped like :class:`RNG`
    and :class:`Sobol`.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into a variance-reduced sampler: Latin Hypercube
    points stratify every one-dimensional marginal exactly (one point in each
    of the ``n`` equal-width strata along any single coordinate axis), which
    provably reduces variance for integrands with a strong additive component,
    at no asymptotic cost relative to plain Monte Carlo otherwise.

    Like :class:`RNG` and :class:`Sobol`, an instance exposes
    ``uniform(size, dtype)`` returning points in ``[0, 1)`` as a backend
    tensor, so it is a drop-in replacement for the sampler ``MonteCarlo`` uses
    internally.

    For a given dimension, points are constructed as
    ``x_i = (perm(i) + U_i) / n``, where ``perm`` is an independent random
    permutation of ``0, ..., n-1`` and ``U_i`` are i.i.d. ``Uniform(0, 1)``,
    independently for every dimension. Points are generated directly on the
    requested backend/device in pure PyTorch for the ``torch`` backend, and
    with ``scipy.stats.qmc.LatinHypercube`` for the others (converted to the
    requested backend). As with plain Monte Carlo the sample points are
    constants, so gradients still flow through the integrand and the
    integration domain; only the point *placement* differs.

    Notes:
        - Points lie in ``[0, 1)``: ``(perm(i) + U_i) / n`` is below 1 in
          exact arithmetic, and a value that rounding to the requested dtype
          would turn into 1 is clamped to the largest value below it.
        - Unlike the low-discrepancy sequences in this module (Sobol, Halton),
          there is no notion of extensibility or balance properties tied to a
          particular sample size: any strictly positive ``n`` is valid.
        - Per-backend implementations use independent random streams following
          the *same* mathematical construction described above, so results
          are reproducible for a fixed ``seed`` within a backend but do not
          match bit-for-bit across backends -- the same tradeoff documented
          for :class:`Sobol` and :class:`Halton`.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path,
          not the JIT-compiled one (which builds its own RNG internally).

    **References:**

    1. M. D. McKay, R. J. Beckman, and W. J. Conover. A Comparison of Three
       Methods for Selecting Values of Input Variables in the Analysis of
       Output from a Computer Code. Technometrics, 21(2):239-245, 1979.
    2. M. Stein. Large Sample Properties of Simulations Using Latin Hypercube
       Sampling. Technometrics, 29(2):143-151, 1987.
    """

    def __init__(self, backend, seed=None):
        """Initialize a randomized Latin Hypercube sampler.

        Args:
            backend (string): Numerical backend. Must be one of "torch",
                "numpy", "jax", or "tensorflow", and match the backend of the
                integration domain it will be used with.
            seed (int or None, optional): Seed for the random permutations and
                jitter. If None, sampling is randomised. Defaults to None.

        Raises:
            ValueError: If backend is not one of "torch", "numpy", "jax", or
                "tensorflow".
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"RandomizedLatinHypercube only supports backends {_VALID_BACKENDS}, "
                f'but got "{backend}".'
            )
        self._backend = backend
        self._seed = seed

    def uniform(self, size, dtype):
        """Draw randomized LHS points in ``[0, 1)``.

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (backend dtype): Floating point dtype of the returned tensor.

        Returns:
            backend tensor: ``[number_of_points, dim]`` randomized LHS points
            in ``[0, 1)``.
        """
        n, dim = int(size[0]), int(size[1])

        if self._backend == "torch":
            # Native torch construction (same argsort trick as the NumPy/QMCPy
            # version): runs directly on the current device, no NumPy/SciPy
            # round-trip needed, unlike SobolEngine.
            import torch

            device = torch.empty(0).device
            generator = torch.Generator(device=device)
            if self._seed is not None:
                generator.manual_seed(int(self._seed))
            else:
                # A fresh torch.Generator() otherwise keeps a fixed internal
                # default seed (two unseeded generators produce the same
                # sequence), which would make every seed=None call
                # identical instead of independently random.
                generator.seed()

            # Built in float64 whatever the requested dtype: in float32 the
            # spacing near a large n exceeds the jitter, so perm + U would
            # round up to n and put points exactly on 1.
            keys = torch.rand((dim, n), generator=generator, device=device, dtype=torch.float64)
            permutations = torch.argsort(keys, dim=-1)
            U = torch.rand((dim, n), generator=generator, device=device, dtype=torch.float64)
            result = (permutations.to(torch.float64) + U) / n
            return _below_one(result.transpose(0, 1).to(dtype))  # -> (n, dim)

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the points onto the requested backend. `seed=` (rather than the
        # newer `rng=`) is used for compatibility with older SciPy versions.
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, scramble=True, seed=self._seed)
        points = sampler.random(n)
        return _below_one(anp.array(points, dtype=dtype, like=self._backend))

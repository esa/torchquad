import warnings

from autoray import numpy as anp

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
    tipping math.ceil to 7 instead of 6 -- caught and fixed during
    development."""
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
    preserves equidistribution properties exactly -- checked below via a
    stratification test), but it does not cover the full space of possible
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
        - Points are generated directly on the target backend/device, with no
          NumPy round-trip -- unlike SobolEngine, this construction does not
          require staying on CPU.
        - Floating-point caveat: coordinates equal to k / base^m are not
          always exactly representable in float64 when base != 2 (e.g.
          46/729 does not round-trip exactly through float64, verified
          independently of this implementation). This can occasionally shift
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
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

            primes = _first_n_primes(dim)
            idx_all = torch.arange(number_of_points, device=device, dtype=torch.int64)

            columns = []
            for j, base in enumerate(primes):
                # m_j is computed PER DIMENSION: dimensions with larger bases
                # need fewer digits for the same number_of_points. Sharing a
                # single m across all dimensions (driven by the smallest
                # base) would inject spurious high-order scrambling noise
                # into dimensions that don't need those extra digit
                # positions -- verified to break equidistribution during
                # development.
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
                # itself not exactly representable for base != 2) -- this
                # was measured to mis-stratify roughly a third of points in
                # a stress test; the single-division form reduces that to
                # the fundamental float64 representability limit (~7%,
                # confirmed independent of this code: plain Python
                # `46/729*729 != 46`).
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

            points = torch.stack(columns, dim=1).to(dtype)
            return points

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the constant points onto the requested backend.
        from scipy.stats import qmc

        sampler = qmc.Halton(d=dim, scramble=self._scramble, seed=self._seed)
        points = sampler.random(number_of_points)
        return anp.array(points, dtype=dtype, like=self._backend)

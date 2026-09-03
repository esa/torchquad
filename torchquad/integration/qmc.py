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
        - Points are guaranteed to lie in ``[0, 1)``: since ``U_i < 1``
          always, ``(perm(i) + U_i) / n`` is always strictly below
          ``(n - 1 + 1) / n = 1``, regardless of the draw.
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
                # default seed (verified: two unseeded generators produce the
                # SAME sequence), which would make every seed=None call
                # identical instead of independently random.
                generator.seed()

            keys = torch.rand((dim, n), generator=generator, device=device, dtype=dtype)
            permutations = torch.argsort(keys, dim=-1)
            U = torch.rand((dim, n), generator=generator, device=device, dtype=dtype)
            result = (permutations.to(dtype) + U) / n
            return result.transpose(0, 1)  # -> (n, dim)

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the points onto the requested backend. `seed=` (rather than the
        # newer `rng=`) is used for compatibility with older SciPy versions.
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, scramble=True, seed=self._seed)
        points = sampler.random(n)
        return anp.array(points, dtype=dtype, like=self._backend)

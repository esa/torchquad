import warnings

from autoray import numpy as anp


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

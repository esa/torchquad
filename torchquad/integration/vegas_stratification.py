from autoray import numpy as anp
from autoray import astype

from .utils import _add_at_indices


class VEGASStratification:
    """The stratification used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, N_increment, dim, rng, backend, dtype, beta=0.75):
        """Initialize the VEGAS stratification.

        Args:
            N_increment (int): Number of evaluations per iteration.
            dim (int): Dimensionality
            rng (RNG): Random number generator
            backend (string): Numerical backend
            dtype (backend dtype): dtype used for the calculations
            beta (float, optional): Beta parameter from VEGAS Enhanced. Defaults to 0.75.
        """
        self.rng = rng
        self.dim = dim
        # stratification steps per dim, EQ 41
        self.N_strat = int((N_increment / 4.0) ** (1.0 / dim))
        self.N_strat = 1000 if self.N_strat > 1000 else self.N_strat
        self.beta = beta  # variable controlling adaptiveness in stratification 0 to 1
        self.N_cubes = self.N_strat**self.dim  # total number of subdomains
        self.V_cubes = (1.0 / self.N_strat) ** self.dim  # volume of hypercubes

        self.dtype = dtype
        self.backend = backend

        # jacobian times f eval and jacobian^2 times f
        self.JF = anp.zeros([self.N_cubes], dtype=self.dtype, like=backend)
        self.JF2 = anp.zeros([self.N_cubes], dtype=self.dtype, like=backend)

        # dampened counts
        self.dh = (
            anp.ones([self.N_cubes], dtype=self.dtype, like=backend)
            * 1.0
            / self.N_cubes
        )

        # current index counts as floating point numbers
        self.strat_counts = anp.zeros([self.N_cubes], dtype=self.dtype, like=backend)

    def accumulate_weight(self, nevals, weight_all_cubes):
        """Accumulate weights for the cubes.

        Args:
            nevals (backend tensor): Number of evals belonging to each cube (sorted).
            weight_all_cubes (backend tensor): Function values.

        Returns:
            backend tensor, backend tensor: Computed JF and JF2
        """
        # indices maps each index of weight_all_cubes to the corresponding
        # hypercube index.
        N_cubes_arange = anp.arange(self.N_cubes, dtype=nevals.dtype, like=self.backend)
        indices = anp.repeat(N_cubes_arange, nevals)
        # Reset JF and JF2, and accumulate the weights and squared weights
        # into them.
        self.JF = anp.zeros([self.N_cubes], dtype=self.dtype, like=self.backend)
        self.JF2 = anp.zeros([self.N_cubes], dtype=self.dtype, like=self.backend)
        _add_at_indices(self.JF, indices, weight_all_cubes, is_sorted=True)
        _add_at_indices(self.JF2, indices, weight_all_cubes**2.0, is_sorted=True)

        # Store counts
        self.strat_counts = astype(nevals, self.dtype)

        return self.JF, self.JF2

    def update_DH(self):
        """Update the dampened sample counts."""
        d_sum = 0
        d_tmp = 0

        # EQ 42
        V2 = self.V_cubes * self.V_cubes
        d_tmp = (
            V2 * self.JF2 / self.strat_counts
            - (self.V_cubes * self.JF / self.strat_counts) ** 2
        )
        # Sometimes rounding errors produce negative values very close to 0
        d_tmp[d_tmp < 0.0] = 0.0

        self.dh = d_tmp**self.beta

        # Normalize dampening
        d_sum = anp.sum(self.dh)
        if d_sum != 0:
            self.dh = self.dh / d_sum

    def get_NH(self, nevals_exp):
        """Recalculate sample points per hypercube, EQ 44.

        Args:
            nevals_exp (int): Expected number of evaluations.

        Returns:
            backend tensor: Stratified sample counts per cube.
        """
        nh = anp.floor(self.dh * nevals_exp)
        nh = anp.clip(nh, 2, None)
        return astype(nh, "int64")

    def _get_indices(self, idx):
        """Maps point to stratified point.

        Args:
            idx (int backend tensor): Target points indices.

        Returns:
            int backend tensor: Mapped points.
        """
        # A commented-out alternative way for mapped points calculation if
        # idx is anp.arange(len(nevals), like=nevals).
        # torch.meshgrid's indexing argument was added in version 1.10.1,
        # so don't use it yet.
        """
        grid_1d = anp.arange(self.N_strat, like=self.backend)
        points = anp.meshgrid(*([grid_1d] * self.dim), indexing="xy", like=self.backend)
        points = anp.stack(
            [mg.ravel() for mg in points], axis=1, like=self.backend
        )
        return points
        """
        # Repeat idx via broadcasting and divide it by self.N_strat ** d
        # for all dimensions d
        points = anp.reshape(idx, [idx.shape[0], 1])
        strides = self.N_strat ** anp.arange(self.dim, like=points)
        if self.backend == "torch":
            # Torch shows a compatibility warning with //, so use torch.div
            # instead
            points = anp.div(points, strides, rounding_mode="floor")
        else:
            points = points // strides
        # Calculate the component-wise remainder: points mod self.N_strat
        points[:, :-1] = points[:, :-1] - self.N_strat * points[:, 1:]
        return points

    def get_Y(self, nevals):
        """Compute randomly sampled points.

        Args:
            nevals (int backend tensor): Number of samples to draw per stratification cube.

        Returns:
            backend tensor: Sampled points.
        """
        # Get integer positions for each hypercube
        nevals_arange = anp.arange(len(nevals), dtype=nevals.dtype, like=nevals)
        positions = self._get_indices(nevals_arange)

        # For each hypercube i, repeat its position nevals[i] times
        position_indices = anp.repeat(nevals_arange, nevals)
        positions = positions[position_indices, :]

        # Convert the positions to float, add random offsets to them and scale
        # the result so that each point is in [0, 1)^dim
        positions = astype(positions, self.dtype)
        random_uni = self.rng.uniform(
            size=[positions.shape[0], self.dim], dtype=self.dtype
        )
        positions = (positions + random_uni) / self.N_strat
        # Due to rounding errors points are sometimes 1.0; replace them with
        # a value close to 1
        positions[positions >= 1.0] = 0.999999
        return positions

from autoray import numpy as anp
from autoray import astype


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
        self.N_cubes = self.N_strat ** self.dim  # total number of subdomains
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

        # current index counts
        self.strat_counts = anp.zeros([self.N_cubes], dtype=self.dtype, like=backend)

    def accumulate_weight(self, nevals, weight_all_cubes):
        """Accumulate weights for the cubes.

        Args:
            nevals (backend tensor): Number of evals belonging to each cube (sorted).
            weight_all_cubes (backend tensor): Function values.

        Returns:
            backend tensor, backend tensor: Computed JF and JF2
        """
        # Build indices for weights
        indices = anp.cumsum(nevals, axis=0)

        # Compute squares of all weights ahead
        square_weights = weight_all_cubes ** 2.0

        # Used to store previous indexstarting point
        prev = 0
        for idx in range(len(nevals)):
            # Get the values for the idx-th cubes
            cur_weights = weight_all_cubes[prev : indices[idx]]
            cur_square_weights = square_weights[prev : indices[idx]]

            # Compute jacobians
            self.JF2[idx] = cur_square_weights.sum()
            self.JF[idx] = cur_weights.sum()

            # Store counts
            self.strat_counts[idx] = len(cur_weights)
            prev = indices[idx]

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

        self.dh = d_tmp ** self.beta

        # for very small d_tmp d_tmp ** self.beta becomes NaN
        self.dh[anp.isnan(self.dh)] = 0

        # Normalize dampening
        d_sum = sum(self.dh)

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
            idx (backend tensor): Target points indices.

        Returns:
            backend tensor: Mapped points.
        """
        # A commented-out alternative way for mapped points calculation.
        # torch.meshgrid's indexing argument was added in version 1.10.1,
        # so don't use it yet.
        """
        grid_1d = torch.arange(self.N_strat)
        points = anp.meshgrid(*([grid_1d] * self.dim), indexing="xy")
        points = torch.stack(
            [mg.ravel() for mg in points], dim=1
        )
        """
        res = anp.zeros([len(idx), self.dim], like=idx)
        tmp = idx
        for i in range(self.dim):
            if self.backend == "torch":
                # Torch shows a compatibility warning with //, so use torch.div
                # instead
                q = anp.div(tmp, self.N_strat, rounding_mode="floor")
            else:
                q = tmp // self.N_strat
            r = tmp - q * self.N_strat
            res[:, i] = r
            tmp = q
        return res

    def get_Y(self, nevals):
        """Compute randomly sampled points.

        Args:
            nevals (int backend tensor): Number of samples to draw per stratification cube.

        Returns:
            int backend tensor: Sampled points.
        """
        dy = 1.0 / self.N_strat
        res_in_all_cubes = []

        # Get indices
        indices = anp.arange(len(nevals), like=nevals)
        indices = astype(self._get_indices(indices), self.dtype)

        # Get random numbers (we get a few more just to vectorize properly)
        # This might increase the memory requirements slightly but is probably
        # worth it.
        random_uni = (
            self.rng.uniform(
                size=[len(nevals), nevals.max(), self.dim], dtype=self.dtype
            )
            * 0.999999
        )

        # Sum the random numbers onto the index locations and scale with dy
        # Note that the resulting tensor is still slightly too large
        # that gets remedied in the for-loop after
        # Also, indices needs the unsqueeye to fill the missing dimension
        indices = indices.reshape(indices.shape[0], 1, indices.shape[1])
        res = (random_uni + indices) * dy

        # Note this loop is tricky to vectorize as cubes have different N
        for idx, N in enumerate(nevals):
            res_in_all_cubes.append(res[idx, 0:N, :])

        return anp.concatenate(res_in_all_cubes, axis=0, like=res)

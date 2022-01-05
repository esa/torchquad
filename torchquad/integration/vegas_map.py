from autoray import numpy as anp
from autoray import astype, infer_backend, to_backend_dtype


class VEGASMap:
    """The map used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation is inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, dim, integration_domain, N_intervals=100, alpha=0.5) -> None:
        """Initializes VEGAS Enhanced's adaptive map

        Args:
            dim (int): Dimensionality of the integrand.
            integration_domain (backend tensor): Integration domain.
            N_intervals (int, optional): Number of intervals to split the domain in. Defaults to 100.
            alpha (float, optional): Alpha from the paper, EQ 19. Defaults to 0.5.
        """
        self.dim = dim
        self.N_intervals = N_intervals  # # of subdivisions
        self.N_edges = self.N_intervals + 1  # # of subdivsion boundaries
        self.alpha = alpha  # Weight smoothing
        self.backend = infer_backend(integration_domain)
        self.dtype = integration_domain.dtype

        # boundary locations and subdomain stepsizes
        self.x_edges = anp.zeros(
            (self.dim, self.N_edges), dtype=self.dtype, like=self.backend
        )
        self.dx_edges = anp.zeros(
            (self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
        )

        # Subdivide initial domain equally spaced in N-d, EQ 8
        for dim in range(self.dim):
            start = integration_domain[dim][0]
            stepsize = (
                integration_domain[dim][1] - integration_domain[dim][0]
            ) / self.N_intervals
            for i in range(self.N_edges):
                self.x_edges[dim][i] = start + i * stepsize
                if i > 0:
                    self.dx_edges[dim][i - 1] = stepsize

        # weights in each intervall
        self.weights = anp.zeros(
            (self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
        )
        self.smoothed_weights = anp.zeros(
            (self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
        )
        self.summed_weights = anp.zeros(
            [self.dim], dtype=self.dtype, like=self.backend
        )  # sum of weights per dim
        self.delta_weights = anp.zeros(
            [self.dim], dtype=self.dtype, like=self.backend
        )  # EQ 11
        self.std_weight = anp.zeros(
            [self.dim], dtype=self.dtype, like=self.backend
        )  # EQ 13
        self.avg_weight = anp.zeros(
            [self.dim], dtype=self.dtype, like=self.backend
        )  # EQ 14
        # numbers of random samples in specific interval
        self.counts = anp.zeros(
            (self.dim, self.N_intervals),
            dtype=to_backend_dtype("int64", like=self.backend),
            like=self.backend,
        )

    def get_X(self, y):
        """Get mapped sampling points, EQ 9.

        Args:
            y (backend tensor): Randomly sampled location(s)

        Returns:
            backend tensor: Mapped points.
        """
        ID, offset = self._get_interval_ID(y), self._get_interval_offset(y)
        res = anp.zeros_like(y)
        for i in range(self.dim):
            ID_i = ID[:, i]
            res[:, i] = self.x_edges[i, ID_i] + self.dx_edges[i, ID_i] * offset[:, i]
        return res

    def get_Jac(self, y):
        """Computes the jacobian of the mapping transformation, EQ 12.

        Args:
            y ([type]): Sampled locations.

        Returns:
            backend tensor: Jacobian
        """
        ID = self._get_interval_ID(y)
        jac = anp.ones([y.shape[0]], dtype=y.dtype, like=y)
        for i in range(self.dim):
            ID_i = ID[:, i]
            jac *= self.N_intervals * self.dx_edges[i][ID_i]
        return jac

    def _get_interval_ID(self, y):
        """Get the integer part of the desired mapping , EQ 10.

        Args:
            y (backend tensor): Sampled points

        Returns:
            backend tensor: Integer part of mapped points.
        """
        return astype(anp.floor(y * float(self.N_intervals)), "int64")

    def _get_interval_offset(self, y):
        """Get the fractional part of the desired mapping , EQ 11.

        Args:
            y (backend tensor): Sampled points.

        Returns:
            backend tensor: Fractional part of mapped points.
        """
        y = y * float(self.N_intervals)
        return y - anp.floor(y)

    def accumulate_weight(self, y, jf_vec2):
        """Accumulate weights and counts of the map.

        Args:
            y (backend tensor): Sampled points.
            jf_vec2 (backend tensor): Square of the product of function values and jacobians
        """
        ID = self._get_interval_ID(y)
        for i in range(self.dim):
            ID_i = ID[:, i]
            unique_vals, unique_counts = anp.unique(ID_i, return_counts=True)
            weights_vals = jf_vec2
            for val in unique_vals:
                self.weights[i][val] += weights_vals[ID_i == val].sum()
            self.counts[i, unique_vals] += unique_counts

    def _smooth_map(self):
        """Smooth the weights in the map, EQ 18 - 22."""
        # EQ 18
        for dim in range(self.dim):
            nnz_idx = self.counts[dim] != 0  # non zero count indices
            self.weights[dim][nnz_idx] = (
                self.weights[dim][nnz_idx] / self.counts[dim][nnz_idx]
            )

        # EQ 18, 19
        for dim in range(self.dim):
            d_sum = sum(self.weights[dim])
            self.summed_weights[dim] = 0

            # i == 0
            d_tmp = (7.0 * self.weights[dim][0] + self.weights[dim][1]) / (8.0 * d_sum)
            d_tmp = pow((d_tmp - 1.0) / anp.log(d_tmp), self.alpha) if d_tmp != 0 else 0
            self.smoothed_weights[dim][0] = d_tmp

            # i == last
            d_tmp = (
                self.weights[dim][self.N_intervals - 2]
                + 7.0 * self.weights[dim][self.N_intervals - 1]
            ) / (8.0 * d_sum)
            d_tmp = pow((d_tmp - 1.0) / anp.log(d_tmp), self.alpha) if d_tmp != 0 else 0
            self.smoothed_weights[dim][-1] = d_tmp

            # rest
            d_tmp = (
                self.weights[dim][:-2]
                + 6.0 * self.weights[dim][1:-1]
                + self.weights[dim][2:]
            ) / (8.0 * d_sum)
            d_tmp[d_tmp != 0] = pow(
                (d_tmp[d_tmp != 0] - 1.0) / anp.log(d_tmp[d_tmp != 0]), self.alpha
            )
            self.smoothed_weights[dim][1:-1] = d_tmp

            # sum all weights
            self.summed_weights[dim] = self.smoothed_weights[dim].sum()

            # EQ 20
            self.delta_weights[dim] = self.summed_weights[dim] / self.N_intervals

    def _reset_weight(
        self,
    ):
        """Resets weights."""
        self.weights = anp.zeros(
            (self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
        )
        self.counts = anp.zeros(
            (self.dim, self.N_intervals),
            dtype=to_backend_dtype("int64", like=self.backend),
            like=self.backend,
        )

    def update_map(
        self,
    ):
        """Update the adaptive map, Section II C."""
        self._smooth_map()

        for i in range(self.dim):  # Update per dim
            old_i = 0
            d_accu = 0

            indices = [0] * (self.N_intervals - 1)
            d_accu_i = anp.zeros(
                [self.N_intervals - 1], dtype=self.dtype, like=self.backend
            )

            for new_i in range(1, self.N_intervals):
                d_accu += self.delta_weights[i]

                while d_accu > self.smoothed_weights[i][old_i]:
                    d_accu -= self.smoothed_weights[i][old_i]
                    old_i = old_i + 1
                indices[new_i - 1] = old_i
                d_accu_i[new_i - 1] = d_accu

            indices = anp.array(
                indices,
                like=self.backend,
                dtype=to_backend_dtype("int64", like=self.backend),
            )

            if self.backend == "torch":
                d_accu_i = d_accu_i.detach()

            # EQ 22
            self.x_edges[i][1:-1] = (
                self.x_edges[i][indices]
                + d_accu_i
                / self.smoothed_weights[i][indices]
                * self.dx_edges[i][indices]
            )

            self.dx_edges[i] = self.x_edges[i][1:] - self.x_edges[i][:-1]

        self._reset_weight()

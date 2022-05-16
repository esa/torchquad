from autoray import numpy as anp
from autoray import astype, to_backend_dtype
from loguru import logger

from .utils import _add_at_indices


class VEGASMap:
    """The map used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation is inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, N_intervals, dim, backend, dtype, alpha=0.5) -> None:
        """Initializes VEGAS Enhanced's adaptive map

        Args:
            N_intervals (int): Number of intervals per dimension to split the domain in.
            dim (int): Dimensionality of the integrand.
            backend (string): Numerical backend
            dtype (backend dtype): dtype used for the calculations
            alpha (float, optional): Alpha from the paper, EQ 19. Defaults to 0.5.
        """
        self.dim = dim
        self.N_intervals = N_intervals  # # of subdivisions
        N_edges = self.N_intervals + 1  # # of subdivsion boundaries
        self.alpha = alpha  # Weight smoothing
        self.backend = backend
        self.dtype = dtype

        # Boundary locations x_edges and subdomain stepsizes dx_edges
        # Subdivide the domain [0,1]^dim equally spaced in N-d, EQ 8
        self.dx_edges = (
            anp.ones((self.dim, self.N_intervals), dtype=self.dtype, like=self.backend)
            / self.N_intervals
        )
        x_edges_per_dim = anp.linspace(
            0.0, 1.0, N_edges, dtype=self.dtype, like=self.backend
        )
        self.x_edges = anp.repeat(
            anp.reshape(x_edges_per_dim, [1, N_edges]), self.dim, axis=0
        )

        # Initialize self.weights and self.counts
        self._reset_weight()

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
        ones = anp.ones(jf_vec2.shape, dtype=self.counts.dtype, like=jf_vec2)
        ID = self._get_interval_ID(y)
        for i in range(self.dim):
            ID_i = ID[:, i]
            _add_at_indices(self.weights[i], ID_i, jf_vec2)
            _add_at_indices(self.counts[i], ID_i, ones)

    @staticmethod
    def _smooth_map(weights, counts, alpha):
        """Smooth the weights in the map, EQ 18 - 22."""
        # Get the average values for J^2 f^2 (weights)
        # EQ 17
        z_idx = counts == 0  # zero count indices
        if anp.any(z_idx):
            nnz_idx = anp.logical_not(z_idx)
            weights[nnz_idx] /= counts[nnz_idx]
            logger.opt(lazy=True).debug(
                "The integrand was not evaluated in {z_idx_sum} of {num_weights} VEGASMap intervals. "
                "Filling the weights for some of them with neighbouring values.",
                z_idx_sum=lambda: anp.sum(z_idx),
                num_weights=lambda: counts.shape[0] * counts.shape[1],
            )
            # Set the weights of the intervals with zero count to weights from
            # their nearest neighbouring intervals
            # (up to a distance of 10 indices).
            for _ in range(10):
                weights[:, :-1] = anp.where(
                    z_idx[:, :-1], weights[:, 1:], weights[:, :-1]
                )
                # The asterisk corresponds to a logical And here
                z_idx[:, :-1] = z_idx[:, :-1] * z_idx[:, 1:]
                weights[:, 1:] = anp.where(
                    z_idx[:, 1:], weights[:, :-1], weights[:, 1:]
                )
                z_idx[:, 1:] = z_idx[:, 1:] * z_idx[:, :-1]
                logger.opt(lazy=True).debug(
                    "  remaining intervals: {z_idx_sum}",
                    z_idx_sum=lambda: anp.sum(z_idx),
                )
                if not anp.any(z_idx):
                    break
        else:
            weights /= counts

        # Convolve with [1/8, 6/8, 1/8] in each dimension to smooth the
        # weights; boundary behaviour: repeat border values.
        # Divide by d_sum to normalize (divide by the sum before smoothing)
        # EQ 18
        dim, N_intervals = weights.shape
        weights_sums = anp.reshape(anp.sum(weights, axis=1), [dim, 1])
        if anp.any(weights_sums == 0.0):
            # The VEGASMap cannot be updated in dimensions where all weights
            # are zero.
            return None
        i_tmp = N_intervals - 2
        d_tmp = anp.concatenate(
            [
                7.0 * weights[:, 0:1] + weights[:, 1:2],
                weights[:, :-2] + 6.0 * weights[:, 1:-1] + weights[:, 2:],
                weights[:, i_tmp : i_tmp + 1] + 7.0 * weights[:, i_tmp + 1 : i_tmp + 2],
            ],
            axis=1,
            like=weights,
        )
        d_tmp = d_tmp / (8.0 * weights_sums)

        # Range compression
        # EQ 19
        d_tmp[d_tmp != 0] = (
            (d_tmp[d_tmp != 0] - 1.0) / anp.log(d_tmp[d_tmp != 0])
        ) ** alpha

        return d_tmp

    def _reset_weight(self):
        """Reset or initialize weights and counts."""
        # weights in each intervall
        self.weights = anp.zeros(
            (self.dim, self.N_intervals), dtype=self.dtype, like=self.backend
        )
        # numbers of random samples in specific interval
        self.counts = anp.zeros(
            (self.dim, self.N_intervals),
            dtype=to_backend_dtype("int64", like=self.backend),
            like=self.backend,
        )

    def update_map(self):
        """Update the adaptive map, Section II C."""
        smoothed_weights = self._smooth_map(self.weights, self.counts, self.alpha)
        if smoothed_weights is None:
            logger.warning(
                "Cannot update the VEGASMap. This can happen with an integrand "
                "which evaluates to zero everywhere."
            )
            self._reset_weight()
            return

        # The amount of the sum of smoothed_weights for each interval of
        # the new 1D grid, for each dimension
        # EQ 20
        delta_weights = anp.sum(smoothed_weights, axis=1) / self.N_intervals

        for i in range(self.dim):  # Update per dim
            delta_d = delta_weights[i]
            # For each inner edge, determine how many delta_d fit into the
            # accumulated smoothed weights.
            # With torch, CUDA and a high number of points the cumsum operation
            # with float32 precision is too inaccurate which leads to wrong
            # indices, so cast to float64 here.
            delta_d_multiples = astype(
                anp.cumsum(astype(smoothed_weights[i, :-1], "float64"), axis=0)
                / delta_d,
                "int64",
            )
            # For each number of delta_d multiples in {0, 1, …, N_intervals},
            # determine how many intervals belong to it (num_sw_per_dw)
            # and the sum of smoothed weights in these intervals (val_sw_per_dw)
            dtype_int = delta_d_multiples.dtype
            num_sw_per_dw = anp.zeros(
                [self.N_intervals + 1], dtype=dtype_int, like=delta_d
            )
            _add_at_indices(
                num_sw_per_dw,
                delta_d_multiples,
                anp.ones(delta_d_multiples.shape, dtype=dtype_int, like=delta_d),
                is_sorted=True,
            )
            val_sw_per_dw = anp.zeros(
                [self.N_intervals + 1], dtype=self.dtype, like=delta_d
            )
            _add_at_indices(
                val_sw_per_dw, delta_d_multiples, smoothed_weights[i], is_sorted=True
            )
            # The cumulative sum of the number of smoothed weights per delta_d
            # multiple determines the old inner edges indices for the new inner
            # edges calculation
            indices = anp.cumsum(num_sw_per_dw[:-2], axis=0)
            # d_accu_i is used for the interpolation in the new inner edges
            # calculation when adding it to the old inner edges
            d_accu_i = anp.cumsum(delta_d - val_sw_per_dw[:-2], axis=0)

            # EQ 22
            self.x_edges[i][1:-1] = (
                self.x_edges[i][indices]
                + d_accu_i / smoothed_weights[i][indices] * self.dx_edges[i][indices]
            )
            finite_edges = anp.isfinite(self.x_edges[i])
            if not anp.all(finite_edges):
                # With float64 precision the delta_d_multiples calculation
                # usually doesn't have rounding errors.
                # If it is nonetheless too inaccurate, few values in
                # smoothed_weights[i][indices] can be zero, which leads to
                # invalid edges.
                num_edges = self.x_edges.shape[1]
                logger.warning(
                    f"{num_edges - anp.sum(finite_edges)} out of {num_edges} calculated VEGASMap edges were infinite"
                )
                # Replace inf edges with the average of their two neighbours
                middle_edges = 0.5 * (self.x_edges[i][:-2] + self.x_edges[i][2:])
                self.x_edges[i][1:-1] = anp.where(
                    finite_edges[1:-1], self.x_edges[i][1:-1], middle_edges
                )
                if not anp.all(anp.isfinite(self.x_edges[i])):
                    raise RuntimeError("Could not replace all infinite edges")

            self.dx_edges[i] = self.x_edges[i][1:] - self.x_edges[i][:-1]

        self._reset_weight()

import torch


class VEGASMap:
    """The map used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation is inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, dim, integration_domain, N_intervals=100, alpha=0.5) -> None:
        """Initializes VEGAS Enhanced's adaptive map

        Args:
            dim (int): Dimensionality of the integrand.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            N_intervals (int, optional): Number of intervals to split the domain in. Defaults to 100.
            alpha (float, optional): Alpha from the paper, EQ 19. Defaults to 0.5.
        """
        self.dim = dim
        self.N_intervals = N_intervals  # # of subdivisions
        self.N_edges = self.N_intervals + 1  # # of subdivsion boundaries
        self.alpha = alpha  # Weight smoothing
        self.x_edges = torch.zeros((self.dim, self.N_edges))  # boundary locations
        self.dx_edges = torch.zeros((self.dim, self.N_intervals))  # subdomain stepsizes

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
        self.weights = torch.zeros((self.dim, self.N_intervals))
        self.smoothed_weights = torch.zeros((self.dim, self.N_intervals))
        self.summed_weights = torch.zeros(self.dim)  # sum of weights per dim
        self.delta_weights = torch.zeros(self.dim)  # EQ 11
        self.std_weight = torch.zeros(self.dim)  # EQ 13
        self.avg_weight = torch.zeros(self.dim)  # EQ 14
        # numbers of random samples in specific interval
        self.counts = torch.zeros((self.dim, self.N_intervals)).long()

    def get_X(self, y):
        """Get mapped sampling points, EQ 9.

        Args:
            y (torch.tensor): Randomly sampled location(s)

        Returns:
            torch.tensor: Mapped points.
        """
        ID, offset = self._get_interval_ID(y), self._get_interval_offset(y)
        res = torch.zeros_like(y)
        for i in range(self.dim):
            ID_i = torch.floor(ID[:, i]).long()
            res[:, i] = self.x_edges[i, ID_i] + self.dx_edges[i, ID_i] * offset[:, i]
        return res

    def get_Jac(self, y):
        """Computes the jacobian of the mapping transformation, EQ 12.

        Args:
            y ([type]): Sampled locations.

        Returns:
            torch.tensor: Jacobian
        """
        ID = self._get_interval_ID(y)
        jac = torch.ones(y.shape[0])
        for i in range(self.dim):
            ID_i = torch.floor(ID[:, i]).long()
            jac *= self.N_intervals * self.dx_edges[i][ID_i]
        return jac

    def _get_interval_ID(self, y):
        """Get the integer part of the desired mapping , EQ 10.

        Args:
            y (float): Sampled point

        Returns:
            int: Integer part of mapped point.
        """
        return torch.floor(y * float(self.N_intervals))

    def _get_interval_offset(self, y):
        """Get the fractional part of the desired mapping , EQ 11.

        Args:
            y (float): Sampled point.

        Returns:
            float: Fractional part of mapped point.
        """
        return (y * self.N_intervals) - self._get_interval_ID(y)

    def accumulate_weight(self, y, jf_vec2):
        """Accumulate weights and counts of the map.

        Args:
            y (float): Sampled point.
            jf_vec2 (float): Square of the product of function value and jacobian
        """
        ID = self._get_interval_ID(y)
        for i in range(self.dim):
            ID_i = torch.floor(ID[:, i]).long()
            unique_vals, unique_counts = torch.unique(ID_i, return_counts=True)
            weights_vals = jf_vec2
            for val in unique_vals:
                self.weights[i][val] += weights_vals[ID_i == val].sum()
            self.counts[i, unique_vals.long()] += unique_counts

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
            d_tmp = (
                pow((d_tmp - 1.0) / torch.log(d_tmp), self.alpha) if d_tmp != 0 else 0
            )
            self.smoothed_weights[dim][0] = d_tmp

            # i == last
            d_tmp = (
                self.weights[dim][self.N_intervals - 2]
                + 7.0 * self.weights[dim][self.N_intervals - 1]
            ) / (8.0 * d_sum)
            d_tmp = (
                pow((d_tmp - 1.0) / torch.log(d_tmp), self.alpha) if d_tmp != 0 else 0
            )
            self.smoothed_weights[dim][-1] = d_tmp

            # rest
            d_tmp = (
                self.weights[dim][:-2]
                + 6.0 * self.weights[dim][1:-1]
                + self.weights[dim][2:]
            ) / (8.0 * d_sum)
            d_tmp[d_tmp != 0] = pow(
                (d_tmp[d_tmp != 0] - 1.0) / torch.log(d_tmp[d_tmp != 0]), self.alpha
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
        self.weights = torch.zeros((self.dim, self.N_intervals))
        self.counts = torch.zeros((self.dim, self.N_intervals)).long()

    def update_map(
        self,
    ):
        """Update the adaptive map, Section II C."""
        self._smooth_map()

        for i in range(self.dim):  # Update per dim
            old_i = 0
            d_accu = 0

            indices = torch.zeros(self.N_intervals - 1).long()
            d_accu_i = torch.zeros(self.N_intervals - 1)

            for new_i in range(1, self.N_intervals):
                d_accu += self.delta_weights[i]

                while d_accu > self.smoothed_weights[i][old_i]:
                    d_accu -= self.smoothed_weights[i][old_i]
                    old_i = old_i + 1
                indices[new_i - 1] = old_i
                d_accu_i[new_i - 1] = d_accu

            # EQ 22
            self.x_edges[i][1:-1] = (
                self.x_edges[i][indices]
                + d_accu_i.detach()
                / self.smoothed_weights[i][indices]
                * self.dx_edges[i][indices]
            )

            self.dx_edges[i] = self.x_edges[i][1:] - self.x_edges[i][:-1]

        self._reset_weight()

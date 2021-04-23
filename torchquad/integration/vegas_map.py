import torch
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)


class VEGASMap:
    """The map used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation is inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, dim, integration_domain, N_intervals=100, alpha=0.5) -> None:
        """Initializes VEGAS Enhanced' adaptive map

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
        self.counts = torch.zeros((self.dim, self.N_intervals))

    def get_X(self, y):
        """Get mapped sampling points, EQ 9.

        Args:
            y (torch.tensor): Randomly sampled location(s)

        Returns:
            torch.tensor: Mapped points.
        """
        ID, offset = self._get_interval_ID(y), self._get_interval_offset(y)
        res = torch.zeros([1, self.dim])
        for i in range(self.dim):
            ID_i = int(ID[i])
            res[0][i] = self.x_edges[i][ID_i] + self.dx_edges[i][ID_i] * offset[i]
        return res

    def get_Jac(self, y):
        """Computes the jacobian of the mapping transformation, EQ 12.

        Args:
            y ([type]): Sampled locations.

        Returns:
            torch.tensor: Jacobian
        """
        ID = self._get_interval_ID(y)
        jac = 1
        for i in range(self.dim):
            ID_i = int(ID[i])
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

    def accumulate_weight(self, y, f):
        """Accumulate weights and counts of the map.

        Args:
            y (float): Sampled point.
            f (float): Function evaluation.
        """
        ID = self._get_interval_ID(y)
        for i in range(self.dim):
            ID_i = int(ID[i])
            self.weights[i][ID_i] += (f * self.get_Jac(y)) ** 2
            self.counts[i][ID_i] += 1

    def _smooth_map(self,):
        """Smooth the weights in the map, EQ 18 - 22.
        """
        # EQ 18
        for i in range(self.dim):
            for i_interval in range(len(self.weights[i])):
                if self.counts[i][i_interval] != 0:
                    self.weights[i][i_interval] = (
                        self.weights[i][i_interval] / self.counts[i][i_interval]
                    )

        # EQ 18, 19
        for dim in range(self.dim):
            d_sum = sum(self.weights[dim])
            self.summed_weights[dim] = 0
            for i in range(self.N_intervals):
                if i == 0:
                    d_tmp = (7.0 * self.weights[dim][0] + self.weights[dim][1]) / (
                        8.0 * d_sum
                    )
                elif i == self.N_intervals - 1:
                    d_tmp = (
                        self.weights[dim][self.N_intervals - 2]
                        + 7.0 * self.weights[dim][self.N_intervals - 1]
                    ) / (8.0 * d_sum)
                else:
                    d_tmp = (
                        self.weights[dim][i - 1]
                        + 6.0 * self.weights[dim][i]
                        + self.weights[dim][i + 1]
                    ) / (8.0 * d_sum)
                if d_tmp != 0:
                    d_tmp = pow((d_tmp - 1.0) / torch.log(d_tmp), self.alpha)

                self.smoothed_weights[dim][i] = d_tmp
                self.summed_weights[dim] += d_tmp

            # EQ 20
            self.delta_weights[dim] = self.summed_weights[dim] / self.N_intervals

    def _reset_weight(self,):
        """Resets weights.
        """
        self.weights = torch.zeros((self.dim, self.N_intervals))
        self.counts = torch.zeros((self.dim, self.N_intervals))

    def update_map(self,):
        """Update the adaptive map, Section II C.
        """
        self._smooth_map()

        # Initialize new locations
        x_edges_last = deepcopy(self.x_edges)
        dx_edges_last = deepcopy(self.dx_edges)

        for i in range(self.dim):  # Update per dim
            new_i = 1
            old_i = 0
            d_accu = 0
            while True:
                d_accu += self.delta_weights[i]

                while d_accu > self.smoothed_weights[i][old_i]:
                    d_accu -= self.smoothed_weights[i][old_i]
                    old_i = old_i + 1

                # EQ 22
                self.x_edges[i][new_i] = (
                    x_edges_last[i][old_i]
                    + d_accu / self.smoothed_weights[i][old_i] * dx_edges_last[i][old_i]
                )

                self.dx_edges[i][new_i - 1] = (
                    self.x_edges[i][new_i] - self.x_edges[i][new_i - 1]
                )

                new_i = new_i + 1
                if new_i >= self.N_intervals:
                    break

            self.dx_edges[i][self.N_intervals - 1] = (
                self.x_edges[i][self.N_edges - 1] - self.x_edges[i][self.N_edges - 2]
            )

        self._reset_weight()


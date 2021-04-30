import torch
import math

import logging

logger = logging.getLogger(__name__)


class VEGASStratification:
    """The stratification used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112 .
    Implementation inspired by https://github.com/ycwu1030/CIGAR/ .
    EQ <n> refers to equation <n> in the above paper.
    """

    def __init__(self, N_increment, dim=1, beta=0.75):
        """Initialize the VEGAS stratification.

        Args:
            N_increment (int, optional): Number of evaluations per iteration.
            dim (int, optional): Dimensionality. Defaults to 1.
            beta (float, optional): Beta parameter from VEGAS Enhanced. Defaults to 0.75.
        """
        self.dim = dim
        # stratification steps per dim, EQ 33
        self.N_strat = math.floor((N_increment / 2.0) ** (1.0 / dim))
        self.beta = beta  # variable controlling adaptiveness in stratification 0 to 1
        self.N_cubes = self.N_strat ** self.dim  # total number of subdomains
        self.V_cubes = (1.0 / self.N_strat) ** self.dim  # volume of hypercubes
        self.JF = torch.zeros(self.N_cubes)  # jacobian times f eval
        self.JF2 = torch.zeros(self.N_cubes)  # jacobian^2 times f
        self.dh = torch.ones(self.N_cubes) * 1.0 / self.N_cubes  # dampened counts
        self.strat_counts = torch.zeros(self.N_cubes)  # current index counts

    def accumulate_weight_vec(self, idx, weight):
        """Accumulate weights for this index + weight.

        Args:
            idx (int): Current index.
            weight (float): Function value.
        """
        self.JF2[idx] += pow(weight, 2).sum()
        self.JF[idx] += weight.sum()
        self.strat_counts[idx] += len(weight)

    def accumulate_weight(self, idx, weight):
        """Accumulate weights for this index + weight.

        Args:
            idx (int): Current index.
            weight (float): Function value.
        """
        self.JF2[idx] += weight * weight
        self.JF[idx] += weight
        self.strat_counts[idx] += 1

    def update_DH(self):
        """Update the dampened sample counts."""
        d_sum = 0
        d_tmp = 0

        for i in range(self.N_cubes):
            # EQ 42
            d_tmp = (
                self.V_cubes * self.V_cubes / self.strat_counts[i] * self.JF2[i]
                - (self.V_cubes / self.strat_counts[i] * self.JF[i]) ** 2
            )

            self.dh[i] = d_tmp ** self.beta  # dampening (EQ 43)

            # for very small d_tmp d_tmp ** self.beta becomes NaN
            if torch.isnan(self.dh[i]):
                self.dh[i] = 0

        # Normalize dampening
        d_sum = sum(self.dh)

        if d_sum != 0:
            self.dh = self.dh / d_sum

    def get_NH(self, idx, nevals_exp):
        """Recalculate sample points per hypercube, EQ 44.

        Args:
            idx (int): Current index.
            nevals_exp (int): Expected number of evaluations.

        Returns:
            int: Stratified sample counts.
        """
        nh = self.dh[idx] * nevals_exp
        if nh < 2:
            return 2
        else:
            return int(nh)

    def _get_indices(self, idx):
        """Maps point to stratified point.

        Args:
            idx (int): Target points index.

        Returns:
            torch.tensor: Mapped point.
        """
        res = torch.zeros([self.dim])
        tmp = idx
        for i in range(self.dim):
            q = tmp // self.N_strat
            r = tmp - q * self.N_strat
            res[i] = r
            tmp = q
        return res

    def get_Y_vec(self, idx, N):
        """Compute randomly sampled points in specified interval.

        Args:
            idx (int): Interval index.

        Returns:
            torch.tensor: Sampled point.
        """
        dy = 1.0 / self.N_strat
        res = torch.zeros([N, self.dim])
        ID = self._get_indices(idx)
        random_uni = torch.rand(size=[N, self.dim])
        res = random_uni * dy + ID * dy
        return res

    def get_Y(self, idx):
        """Compute randomly sampled points in specified interval.

        Args:
            idx (int): Interval index.

        Returns:
            torch.tensor: Sampled point.
        """
        dy = 1.0 / self.N_strat
        res = torch.zeros([self.dim])
        ID = self._get_indices(idx)
        random_uni = torch.rand(size=[self.dim])
        for i in range(self.dim):
            res[i] = random_uni[i] * dy + ID[i] * dy
        return res

import torch
import math


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
        self.N_strat = 1000 if self.N_strat > 1000 else self.N_strat
        self.beta = beta  # variable controlling adaptiveness in stratification 0 to 1
        self.N_cubes = self.N_strat ** self.dim  # total number of subdomains
        self.V_cubes = (1.0 / self.N_strat) ** self.dim  # volume of hypercubes
        self.JF = torch.zeros(self.N_cubes)  # jacobian times f eval
        self.JF2 = torch.zeros(self.N_cubes)  # jacobian^2 times f
        self.dh = torch.ones(self.N_cubes) * 1.0 / self.N_cubes  # dampened counts
        self.strat_counts = torch.zeros(self.N_cubes)  # current index counts

    def accumulate_weight(self, nevals, weight_all_cubes):
        """Accumulate weights for the cubes.

        Args:
            nevals (torch.tensor): Number of evals belonging to each cube (sorted).
            weight_all_cubes (torch.tensor): Function values.

        Returns:
            torch.tensor,torch.tensor: Computed JF and JF2
        """
        # Build indices for weights
        indices = torch.cumsum(nevals, dim=0)

        # Compute squares of all weights ahead
        square_weights = torch.pow(weight_all_cubes, 2.0)

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
            self.strat_counts[idx] += len(cur_weights)
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
        self.dh[torch.isnan(self.dh)] = 0

        # Normalize dampening
        d_sum = sum(self.dh)

        if d_sum != 0:
            self.dh = self.dh / d_sum

    def get_NH(self, nevals_exp):
        """Recalculate sample points per hypercube, EQ 44.

        Args:
            nevals_exp (int): Expected number of evaluations.

        Returns:
            torch.tensor: Stratified sample counts per cube.
        """
        nh = torch.multiply(self.dh, nevals_exp)
        nh = torch.floor(nh)
        nh = torch.clip(nh, 2, None).int()
        return nh

    def _get_indices(self, idx):
        """Maps point to stratified point.

        Args:
            idx (int): Target points index.

        Returns:
            torch.tensor: Mapped point.
        """
        res = torch.zeros([len(idx), self.dim])
        tmp = idx
        for i in range(self.dim):
            q = tmp // self.N_strat
            r = tmp - q * self.N_strat
            res[:, i] = r
            tmp = q
        return res

    def get_Y(self, nevals):
        """Compute randomly sampled points.

        Args:
            nevals (torch.tensor): Number of samples to draw per stratification cube.

        Returns:
            torch.tensor: Sampled points.
        """
        dy = 1.0 / self.N_strat
        res_in_all_cubes = []

        # Get indices
        indices = torch.arange(len(nevals))
        indices = self._get_indices(indices)

        # Get random numbers (we get a few more just to vectorize properly)
        # This might increase the memory requirements slightly but is probably
        # worth it.
        random_uni = torch.rand(size=[len(nevals), nevals.max(), self.dim]) * 0.999999

        # Sum the random numbers onto the index locations and scale with dy
        # Note that the resulting tensor is still slightly too large
        # that gets remedied in the for-loop after
        # Also, indices needs the unsqueeye to fill the missing dimension
        res = (random_uni + indices.unsqueeze(1)) * dy

        # Note this loop is tricky to vectorize as cubes have different N
        for idx, N in enumerate(nevals):
            res_in_all_cubes.append(res[idx, 0:N, :])

        return torch.cat(res_in_all_cubes, dim=0)

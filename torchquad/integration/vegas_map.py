import torch

import logging

logger = logging.getLogger(__name__)


class VEGASMap:
    """The map used for vegas enhanced. Refer to https://arxiv.org/abs/2009.05112.
    """

    def __init__(self) -> None:
        pass

    def get_X(self, y):
        pass

    def get_Jac(self, y):
        pass

    def _get_interval_ID(self, y):
        return torch.floor(y * self.N_intervals)

    def _get_interval_offset(self, y):
        return (y * self.N_intervals) - self._get_interval_ID(y)

    def map_accumulate_weight(self, y, f):
        pass

    def _smooth_map(self,):
        pass

    def _reset_weight(self,):
        pass

    def update_map(self,):
        pass


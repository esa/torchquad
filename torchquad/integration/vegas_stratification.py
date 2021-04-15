import torch

import logging

logger = logging.getLogger(__name__)


class VEGASStratification:
    """The stratification used for VEGAS Enhanced. Refer to https://arxiv.org/abs/2009.05112.
    """

    def __init__(self) -> None:
        pass

    def strat_accumulate_weight(self, idx, weight):
        pass

    def update_DH(self):
        pass

    def get_NH(self, idx, nevals_exp):
        pass

    def strat_get_indices(self, idx):
        pass

    def get_Y(self, idx):
        pass


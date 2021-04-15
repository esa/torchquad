from .base_integrator import BaseIntegrator
from .utils import setup_integration_domain

from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification

import torch

import logging

logger = logging.getLogger(__name__)


class VEGAS(BaseIntegrator):
    """VEGAS Enhanced in torch. Refer to https://arxiv.org/abs/2009.05112."""

    def __init__(self):
        super().__init__()

    def integrate(
        self,
        fn,
        dim,
        N=1000,
        integration_domain=None,
        seed=None,
        use_grid_improve=True,
        eps_rel=0,
        eps_abs=0,
    ):
        """Integrates the passed function on the passed domain using VEGAS

        Args:
            fn (func): The function to integrate over
            dim (int): Dimensionality of the function to integrate
            N (int, optional): Maximum number of function evals to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.
            use_grid_improve (bool, optional): If True will improve the grid after each iteration.
            eps_rel (float, optional): Relative error to abort at. Defaults to 0.
            eps_abs (float, optional): Absolute error to abort at. Defaults to 0.

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            float: Integral value
        """

        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "VEGAS integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points over "
            + str(integration_domain),
        )

        self._dim = dim
        self._nr_of_fevals = 0
        self._max_iterations = 10
        # TODO think about including warmup and grid improvement in this
        self._starting_N = N // self._max_iterations
        self._N_increment = N // self._max_iterations
        self.fn = fn
        self._integration_domain = setup_integration_domain(dim, integration_domain)
        if seed is not None:
            torch.random.manual_seed(seed)

        self.map = VEGASMap()
        self.strat = VEGASStratification()

        logger.debug("Running grid warmup")
        self._improve_grid()

        logger.debug("Running VEGAS Iterations")
        self.results = []  # contains integrations' results per iteration
        self.sigma2 = []  # contains variance per iteration

        yrnd = torch.zeros(self._dim)
        y = torch.zeros(self._dim)
        x = torch.zeros(self._dim)

        while it < self._max_iterations:
            it = it + 1
            self.results.append(0)
            self.sigma2.append(0)

            acc = self._run_iteration(self)

            if it % 5 == 0:
                res = self.get_result()
                err = self.get_error()
                chi2 = self.get_chisq()
                acc = err / res

                print(f"Chi2={chi2[0]:.4e}")
                if (acc < eps_rel or err < eps_abs) and chi2 / 5.0 < 1.0:
                    break
                if chi2 / 5.0 < 1.0:
                    self._starting_N = torch.minimum(
                        self._starting_N + self._N_increment,
                        self._starting_N * torch.sqrt(acc / eps_rel),
                    )
                    self.results = []
                    self.sigma2 = []
                    continue
                elif chi2 / 5.0 > 1.0:
                    self._starting_N += self._N_increment
                    self.results = []
                    self.sigma2 = []
                    continue

            logging.debug(
                f"Iteration {it}, Acc={acc[0]:.4e}, Result={self.results[-1][0]:.4e},neval={self._nr_of_fevals}"
            )

        logger.info("Computed integral was " + str(self.results[-1][0]) + ".")
        return -1

    def _improve_grid(self):
        pass

    def _run_iteration(self):
        # for i_cube in range(vegas.N_cubes):
        #     jf = 0
        #     jf2 = 0
        #     neval = vegas.get_NH(i_cube, neval_start)
        #     self._nr_of_fevals += neval
        #     for ne in range(neval):
        #         y = vegas.get_Y(i_cube)
        #         x = vegas.get_X(y)
        #         f_eval = func(x)
        #         jac = vegas.get_Jac(y)
        #         if f_eval is None or jac is None:
        #             ne = ne - 1
        #             continue
        #         vegas.strat_accumulate_weight(i_cube, f_eval * jac)
        #         jf += f_eval * jac
        #         jf2 += pow(f_eval * jac, 2)
        #     ih = jf / neval * dV
        #     sig2 = jf2 / neval * dV * dV - pow(jf / neval * dV, 2)
        #     self.results[-1] += ih
        #     self.sigma2[-1] += sig2 / neval

        # vegas.update_DH()
        # acc = torch.sqrt(self.sigma2[-1] / self.results[-1])

        # return acc
        pass

    # Helper funcs
    def _get_result(self):
        res_num = 0
        res_den = 0
        for idx, res in enumerate(self.results):
            res_num += res / self.sigma2[idx]
            res_den += 1.0 / self.sigma2[idx]
        return res_num / res_den

    def _get_error(self):
        res = 0
        for sig in self.sigma2:
            res += 1.0 / sig
        return 1.0 / torch.sqrt(res)

    def _get_chisq(self):
        I_final = self.get_result(self.results, self.sigma2)
        chi2 = 0
        for idx, res in enumerate(self.results):
            chi2 += pow(res - I_final, 2) / self.sigma2[idx]
        return chi2

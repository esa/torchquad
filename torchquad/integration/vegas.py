from .base_integrator import BaseIntegrator
from .utils import setup_integration_domain

from .vegas_map import VEGASMap
from .vegas_stratification import VEGASStratification

import torch

import logging

logger = logging.getLogger(__name__)


class VEGAS(BaseIntegrator):
    """Vegas Enhanced in torch. Refer to https://arxiv.org/abs/2009.05112.
    Implementation inspired by https://github.com/ycwu1030/CIGAR/ .
    """

    def __init__(self):
        super().__init__()
        logger.setLevel(logging.DEBUG)

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
        self._fn = fn
        self._integration_domain = setup_integration_domain(dim, integration_domain)
        if seed is not None:
            torch.random.manual_seed(seed)

        self.map = VEGASMap(self._dim, self._integration_domain)
        self.strat = VEGASStratification(self._dim)

        logger.debug("Starting VEGAS")
        self.results = []  # contains integrations results per iteration
        self.sigma2 = []  # contains variance per iteration

        it = 0
        while it < self._max_iterations:
            if use_grid_improve:
                logger.debug("Running grid improvement")
                self._improve_grid()

            it = it + 1
            self.results.append(0)
            self.sigma2.append(0)

            logger.debug("Running VEGAS Iteration")
            acc = self._run_iteration()

            if it % 5 == 0:
                res = self._get_result()
                err = self._get_error()
                chi2 = self._get_chisq()
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
                f"Iteration {it}, Acc={acc:.4e}, Result={self.results[-1]:.4e},neval={self._nr_of_fevals}"
            )

        logger.info("Computed integral was " + str(self.results[-1][0]) + ".")
        return -1

    def _improve_grid(self):
        yrnd = torch.zeros(self._dim)
        y = torch.zeros(self._dim)
        x = torch.zeros(self._dim)
        neval_start = 1000
        alpha_start = 0.5
        self.alpha = alpha_start
        dV = self.strat.V_cubes
        # Warmup
        logger.debug(
            "|  Iter  |    N_Eval    |     Result     |      Error     |    Acc        |"
        )
        for warmup_iter in range(5):
            self.results.append(0)
            self.sigma2.append(0)
            jf = 0
            jf2 = 0

            for ne in range(neval_start):
                yrnd = torch.rand(size=[self._dim])
                x = self.map.get_X(yrnd)
                f_eval = self._eval(x)[0]
                jac = self.map.get_Jac(yrnd)
                if f_eval is None or jac is None:
                    ne = ne - 1
                    continue
                self.map.accumulate_weight(yrnd, f_eval)
                jf += f_eval * jac
                jf2 += pow(f_eval * jac, 2)
            ih = jf / neval_start
            sig2 = jf2 / neval_start - pow(jf / neval_start, 2)
            self.results[-1] += ih
            self.sigma2[-1] += sig2 / neval_start
            self.map.update_map()
            acc = torch.sqrt(self.sigma2[-1] / self.results[-1])
            logger.debug(
                f"|\t{warmup_iter}|         {neval_start}|  {self.results[-1]:5e}  |  {self.sigma2[-1]:5e}  |  {acc:4e}%|"
            )

        res = self._get_result()
        err = self._get_error()
        acc = err / res

        self.results.clear()
        self.sigma2.clear()

        it = 0
        logger.debug(
            "|  Iter  |    N_Eval    |     Result     |      Error     |    Acc        |"
        )
        while True:
            it += 1
            self.results.append(0)
            self.sigma2.append(0)
            for i_cube in range(self.strat.N_cubes):
                jf = 0
                jf2 = 0
                neval = self.strat.get_NH(i_cube, neval_start)
                for ne in range(neval):
                    y = self.strat.get_Y(i_cube)
                    x = self.map.get_X(y)
                    f_eval = self._eval(x)[0]
                    jac = self.map.get_Jac(y)
                    if f_eval is None or jac is None:
                        ne = ne - 1
                        continue
                    self.map.accumulate_weight(y, f_eval)
                    self.strat.accumulate_weight(i_cube, f_eval * jac)
                    jf += f_eval * jac
                    jf2 += pow(f_eval * jac, 2)
                ih = jf / neval * dV
                sig2 = jf2 / neval * dV * dV - pow(jf / neval * dV, 2)
                self.results[-1] += ih
                self.sigma2[-1] += sig2 / neval
            if it % 2 == 1:
                self.map.update_map()
            else:
                self.strat.update_DH()
            acc = torch.sqrt(self.sigma2[-1] / self.results[-1])
            logger.debug(
                f"|\t{it}|         {neval_start}|  {self.results[-1]:5e}  |  {self.sigma2[-1]:5e}  |  {acc:4e}%|"
            )
            if it % 5 == 0:
                res = self._get_result()
                err = self._get_error()
                acc = err / res

                if acc < 0.01:
                    break
                neval_start = neval_start * torch.sqrt(acc / 0.01)
                self.results = []
                self.sigma2 = []

    def _run_iteration(self):
        yrnd = torch.zeros(self._dim)
        y = torch.zeros(self._dim)
        x = torch.zeros(self._dim)

        for i_cube in range(self.strat.N_cubes):
            jf = 0
            jf2 = 0
            neval = self.strat.get_NH(i_cube, self._starting_N)
            self._nr_of_fevals += neval
            for ne in range(neval):
                y = self.strat.get_Y(i_cube)
                x = self.map.get_X(y)
                f_eval = self._eval(x)[0]
                jac = self.map.get_Jac(y)
                if f_eval is None or jac is None:
                    ne = ne - 1
                    continue
                self.strat.accumulate_weight(i_cube, f_eval * jac)
                jf += f_eval * jac
                jf2 += pow(f_eval * jac, 2)
            ih = jf / neval * self.strat.V_cubes
            sig2 = jf2 / neval * self.strat.V_cubes * self.strat.V_cubes - pow(
                jf / neval * self.strat.V_cubes, 2
            )
            self.results[-1] += ih
            self.sigma2[-1] += sig2 / neval

        self.strat.update_DH()
        acc = torch.sqrt(self.sigma2[-1] / self.results[-1])

        return acc

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
        I_final = self._get_result()
        chi2 = 0
        for idx, res in enumerate(self.results):
            chi2 += pow(res - I_final, 2) / self.sigma2[idx]
        return chi2

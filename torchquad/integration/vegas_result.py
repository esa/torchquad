from dataclasses import dataclass
from typing import Any, Optional


def goodness_of_fit_q(chi2, dof):
    """Compute the VEGAS goodness-of-fit p-value Q.

    Q is the probability that a chi-squared statistic with ``dof`` degrees of
    freedom exceeds the observed ``chi2`` purely by chance. Values close to 1
    indicate the per-iteration estimates are consistent with each other; values
    close to 0 indicate the error estimate should not be trusted.

    Args:
        chi2 (backend scalar): Observed chi-squared of the iteration estimates.
        dof (int): Degrees of freedom (combined iterations minus one).

    Returns:
        float or None: The p-value in [0, 1], or None if ``dof`` is not positive
        (a single iteration carries no goodness-of-fit information).
    """
    if dof <= 0:
        return None
    # Regularized upper incomplete gamma = survival function of the chi-squared
    # distribution. SciPy is already a hard dependency.
    from scipy.special import gammaincc

    return float(gammaincc(dof / 2.0, float(chi2) / 2.0))


@dataclass
class VEGASResult:
    """A VEGAS integration result bundled with its error estimate.

    Returned by :meth:`VEGAS.integrate` when ``return_error=True``. The
    tensor-valued fields keep the numerical backend of the integration. Only
    ``integral`` is on the gradient path; ``sdev`` and ``chi2`` are derived from
    the per-iteration variances, which VEGAS detaches, so they are not
    differentiable.
    """

    integral: Any
    """backend scalar: Estimated integral (weighted mean over combined iterations)."""

    sdev: Any
    """backend scalar: Estimated 1-sigma standard deviation of the integral."""

    chi2: Any
    """backend scalar: Chi-squared of the per-iteration estimates about their weighted mean."""

    dof: int
    """int: Degrees of freedom, i.e. the number of combined iterations minus one."""

    Q: Optional[float]
    """float or None: Goodness-of-fit p-value ``P(chi2 >= observed | dof)``; None if ``dof <= 0``."""

    nr_of_fevals: int
    """int: Number of integrand evaluations performed."""

    def __repr__(self):
        """Return a compact ``value +/- sdev (chi2/dof = ..., Q = ...)`` summary."""
        chi2_over_dof = float(self.chi2) / self.dof if self.dof > 0 else float("nan")
        q_text = "n/a" if self.Q is None else f"{self.Q:.2g}"
        return (
            f"VEGASResult({float(self.integral):.6g} +/- {float(self.sdev):.2g}, "
            f"chi2/dof = {chi2_over_dof:.2g}, Q = {q_text})"
        )

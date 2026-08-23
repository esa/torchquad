#!/usr/bin/env python3
"""Generate the figures used in the README's Performance section.

Separate from ``modular_benchmark.py`` on purpose. That harness answers "how
fast is this on my hardware"; these plots answer the question a prospective user
actually asks first -- *how accurate is it, and how does that scale* -- and the
two need different ground rules:

- **float64 throughout.** The existing plots are float32, which floors the error
  near 1e-7 and hides the difference between Simpson, Boole and Gauss-Legendre
  entirely.
- **Genz integrands, not the analytic test suite.** See ``genz_functions`` for
  why: the correctness suite is polynomial-heavy and separable, which makes
  Gauss-Legendre exact and flatters QMC.
- **Error against a closed form**, never against a high-N run of the same
  library, so no method is scored against its own output.

Usage:
    python benchmarking/release_plots.py --plots all
    python benchmarking/release_plots.py --plots convergence,qmc
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchquad import (  # noqa: E402
    Boole,
    GaussLegendre,
    MonteCarlo,
    Simpson,
    Sobol,
    Trapezoid,
    set_up_backend,
)

# Sibling module; this script is run as `python benchmarking/release_plots.py`,
# which puts benchmarking/ on sys.path.
from genz_functions import GENZ_FAMILY  # noqa: E402
from timing import time_integration  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "resources"

# Deterministic rules need N to factor as a per-dimension grid, so the counts
# below are chosen per dimension rather than shared with the stochastic methods.
_DETERMINISTIC = {
    "Trapezoid": Trapezoid,
    "Simpson": Simpson,
    "Boole": Boole,
    "GaussLegendre": GaussLegendre,
}


def _hardware_label(precision="float64"):
    """Describe the machine and precision the numbers were taken at.

    Args:
        precision (str, optional): Floating point precision the measurement used.
            Passed explicitly rather than assumed, because the accuracy plots run
            in float64 while the runtime plot runs in float32, and a figure that
            names the wrong one is worse than one that names none.

    Returns:
        str: One-line hardware and version summary for the figure subtitle.
    """
    device = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else platform.processor() or "CPU"
    )
    return f"{device} | torch {torch.__version__} | {precision}"


def _grid_counts(dim, per_dimension):
    """Total point counts for a tensor-product grid.

    Args:
        dim (int): Dimensionality.
        per_dimension (list): Points along each axis.

    Returns:
        list: Total point counts, one per entry of ``per_dimension``.
    """
    return [n**dim for n in per_dimension]


def _integrate(integrator, function, total_points, **kwargs):
    """Run one integration and return its relative error and wall-clock time.

    Args:
        integrator: A torchquad integrator instance.
        function (GenzFunction): Integrand with a closed-form integral.
        total_points (int): Number of function evaluations to request.
        **kwargs: Extra arguments forwarded to ``integrate()``.

    Returns:
        tuple: ``(relative_error, elapsed_seconds)``, with the error floored at
        1e-16 so it stays plottable on a log axis.
    """
    elapsed, value = time_integration(
        lambda: integrator.integrate(
            function,
            dim=function.dim,
            N=total_points,
            integration_domain=function.integration_domain,
            **kwargs,
        )
    )
    return max(function.relative_error(value), 1e-16), elapsed


def plot_convergence(dim=3, save=True):
    """Plot relative error against evaluation count for every method.

    This is the figure the Performance section is missing: for a numerical
    integration library it is the first thing a reader looks for.

    Args:
        dim (int, optional): Dimensionality. Defaults to 3.
        save (bool, optional): Write the figure and its data. Defaults to True.

    Returns:
        dict: The measured series, keyed by method name.
    """
    function = GENZ_FAMILY["oscillatory"](dim, a=3.0)
    results = {}

    per_dimension = [4, 6, 9, 13, 19, 27, 39, 55]
    for name, integrator_class in _DETERMINISTIC.items():
        integrator = integrator_class()
        counts, errors = [], []
        for total in _grid_counts(dim, per_dimension):
            try:
                error, _ = _integrate(integrator, function, total)
            except Exception as exc:  # noqa: BLE001 - report and continue, do not fake a point
                print(f"  {name} N={total}: skipped ({type(exc).__name__}: {exc})")
                continue
            counts.append(total)
            errors.append(error)
        results[name] = {"n": counts, "error": errors}
        print(f"  {name:15s} {len(counts)} points, best {min(errors):.2e}")

    stochastic_counts = [2**k for k in range(10, 19)]
    for name, rng_factory in (
        ("MonteCarlo", None),
        ("MonteCarlo + Sobol", lambda seed: Sobol("torch", seed=seed)),
    ):
        integrator = MonteCarlo()
        errors = []
        for total in stochastic_counts:
            # Average over seeds: one draw is a sample from a distribution, not
            # a convergence rate, and a single lucky cancellation would read as
            # an order of magnitude of accuracy that is not there.
            trials = []
            for seed in range(5):
                kwargs = {"seed": seed} if rng_factory is None else {"rng": rng_factory(seed)}
                error, _ = _integrate(integrator, function, total, **kwargs)
                trials.append(error)
            errors.append(float(np.median(trials)))
        results[name] = {"n": stochastic_counts, "error": errors}
        print(f"  {name:15s} {len(stochastic_counts)} points, best {min(errors):.2e}")

    if save:
        _draw_convergence(results, function, dim)
    return results


def _draw_convergence(results, function, dim):
    """Render the convergence figure.

    Args:
        results (dict): Series keyed by method name.
        function (GenzFunction): The integrand used.
        dim (int): Dimensionality.
    """
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for name, series in results.items():
        if not series["n"]:
            continue
        style = "--o" if "Monte" in name or "Sobol" in name else "-o"
        axis.loglog(series["n"], series["error"], style, label=name, markersize=4)

    reference = np.array([2**10, 2**18], dtype=float)
    axis.loglog(
        reference, 3e-2 * (reference / reference[0]) ** -0.5, "k:", lw=1, label="$O(N^{-1/2})$"
    )
    axis.loglog(
        reference, 3e-2 * (reference / reference[0]) ** -1.0, "k--", lw=1, label="$O(N^{-1})$"
    )

    axis.set_xlabel("Number of function evaluations $N$")
    axis.set_ylabel("Relative error")
    axis.set_title(
        f"Convergence at {dim}D on the Genz {function.name.lower()} integrand\n{_hardware_label()}",
        fontsize=10,
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save(figure, results, f"torchquad_convergence_{dim}d")


def plot_qmc_vs_mc(dim=3, save=True):
    """Plot quasi-Monte Carlo against plain Monte Carlo convergence.

    The most relevant new figure for 0.6, which ships the Sobol sampler.

    Args:
        dim (int, optional): Dimensionality. Defaults to 3.
        save (bool, optional): Write the figure and its data. Defaults to True.

    Returns:
        dict: The measured series, keyed by sampler name.
    """
    function = GENZ_FAMILY["product_peak"](dim, a=4.0)
    counts = [2**k for k in range(10, 19)]
    results = {}

    for name, rng_factory in (
        ("Monte Carlo (pseudo-random)", None),
        ("QMC: Sobol (scrambled)", lambda seed: Sobol("torch", seed=seed)),
    ):
        errors = []
        for total in counts:
            trials = []
            for seed in range(5):
                kwargs = {"seed": seed} if rng_factory is None else {"rng": rng_factory(seed)}
                error, _ = _integrate(MonteCarlo(), function, total, **kwargs)
                trials.append(error)
            errors.append(float(np.median(trials)))
        results[name] = {"n": counts, "error": errors}
        print(f"  {name:28s} best {min(errors):.2e}")

    if save:
        figure, axis = plt.subplots(figsize=(8, 5.5))
        for name, series in results.items():
            axis.loglog(series["n"], series["error"], "-o", label=name, markersize=5)
        reference = np.array([counts[0], counts[-1]], dtype=float)
        base = results["Monte Carlo (pseudo-random)"]["error"][0]
        axis.loglog(
            reference, base * (reference / reference[0]) ** -0.5, "k:", lw=1, label="$O(N^{-1/2})$"
        )
        axis.loglog(
            reference, base * (reference / reference[0]) ** -1.0, "k--", lw=1, label="$O(N^{-1})$"
        )
        axis.set_xlabel("Number of function evaluations $N$")
        axis.set_ylabel("Median relative error over 5 seeds")
        axis.set_title(
            f"Quasi-Monte Carlo vs Monte Carlo at {dim}D, Genz product peak\n{_hardware_label()}",
            fontsize=10,
        )
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=9)
        figure.tight_layout()
        _save(figure, results, f"torchquad_qmc_vs_mc_{dim}d")
    return results


def plot_dimension_scaling(save=True):
    """Plot error against dimension at a fixed evaluation budget.

    This is where the README's "withstanding the curse of dimensionality" goal
    gets a picture. Difficulty is held constant per dimension, so the curve shows
    the cost of dimension rather than the cost of a harder integrand.

    Args:
        save (bool, optional): Write the figure and its data. Defaults to True.

    Returns:
        dict: The measured series, keyed by method name.
    """
    dimensions = [1, 2, 3, 4, 5, 6, 8, 10]
    budget = 2**16
    results = {
        "MonteCarlo": {"dim": [], "error": []},
        "MonteCarlo + Sobol": {"dim": [], "error": []},
    }

    for dim in dimensions:
        function = GENZ_FAMILY["gaussian"](dim, a=2.0)
        for name, rng_factory in (
            ("MonteCarlo", None),
            ("MonteCarlo + Sobol", lambda seed: Sobol("torch", seed=seed)),
        ):
            trials = []
            for seed in range(5):
                kwargs = {"seed": seed} if rng_factory is None else {"rng": rng_factory(seed)}
                error, _ = _integrate(MonteCarlo(), function, budget, **kwargs)
                trials.append(error)
            results[name]["dim"].append(dim)
            results[name]["error"].append(float(np.median(trials)))
        print(
            f"  d={dim:2d}  MC {results['MonteCarlo']['error'][-1]:.2e}  "
            f"Sobol {results['MonteCarlo + Sobol']['error'][-1]:.2e}"
        )

    if save:
        figure, axis = plt.subplots(figsize=(8, 5.5))
        for name, series in results.items():
            axis.semilogy(series["dim"], series["error"], "-o", label=name, markersize=5)
        axis.set_xlabel("Dimension $d$")
        axis.set_ylabel("Median relative error over 5 seeds")
        axis.set_title(
            f"Error vs dimension at a fixed budget of $N=2^{{16}}$, Genz Gaussian\n"
            f"{_hardware_label()}",
            fontsize=10,
        )
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=9)
        figure.tight_layout()
        _save(figure, results, "torchquad_dimension_scaling")
    return results


def _save(figure, data, stem, precision="float64"):
    """Write a figure and the data behind it.

    The JSON goes next to the PNG so the figure can be redrawn without re-running
    the benchmark, and so a regeneration is diffable.

    Args:
        figure (matplotlib.figure.Figure): The figure to write.
        data (dict): The measured series behind it.
        stem (str): File name without extension.
        precision (str, optional): Precision the measurement used, recorded in
            the JSON alongside the hardware. Defaults to "float64".
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    payload = {"hardware": _hardware_label(precision), "results": data}
    with open(OUTPUT_DIR / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"  wrote {stem}.png and {stem}.json")


def _runtime_worker(device):
    """Measure runtime against N for one device and print the result as JSON.

    Run in its own process because ``set_up_backend`` configures torch's global
    default device: a CPU and a GPU measurement cannot coexist in one
    interpreter, and attempting it silently runs one of them on the wrong
    device rather than failing.

    Args:
        device (str): Either "cpu" or "gpu".
    """
    import os

    if device == "cpu":
        # Must happen before torch initializes CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import torch as torch_module

    from torchquad import MonteCarlo, Simpson, set_up_backend

    set_up_backend("torch", data_type="float32")

    counts = [10**k for k in range(4, 9)]
    measurements = {}
    for name, integrator, kwargs in (
        ("MonteCarlo", MonteCarlo(), {"seed": 0}),
        ("Simpson", Simpson(), {}),
    ):
        times = []
        for total in counts:

            def call(total=total, integrator=integrator, kwargs=kwargs):
                return integrator.integrate(
                    lambda x: torch_module.sin(x[:, 0]),
                    dim=1,
                    N=total,
                    integration_domain=[[0.0, 3.141592653589793]],
                    **kwargs,
                )

            # One discarded warm-up, then the median of three, so a single
            # scheduling hiccup does not become a data point.
            time_integration(call)
            runs = [time_integration(call)[0] for _ in range(3)]
            times.append(float(np.median(runs)))
        measurements[name] = times

    print(json.dumps({"device": device, "n": counts, "times": measurements}))


def plot_runtime_cpu_vs_gpu(save=True):
    """Plot wall-clock runtime against N on CPU and GPU.

    Shows the crossover honestly: below roughly 1e5 evaluations the GPU loses to
    kernel-launch overhead, and saying so is more persuasive than a flat claim
    of acceleration.

    Args:
        save (bool, optional): Write the figure and its data. Defaults to True.

    Returns:
        dict: Measurements keyed by device.
    """
    import subprocess

    measurements = {}
    for device in ("gpu", "cpu"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-runtime", device],
            capture_output=True,
            text=True,
            check=True,
        )
        measurements[device] = json.loads(completed.stdout.strip().splitlines()[-1])
        print(f"  {device}: measured {len(measurements[device]['n'])} point counts")

    if save:
        figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
        counts = measurements["gpu"]["n"]
        for method in measurements["gpu"]["times"]:
            left.loglog(
                counts,
                measurements["gpu"]["times"][method],
                "-o",
                label=f"{method}, GPU",
                markersize=4,
            )
            left.loglog(
                counts,
                measurements["cpu"]["times"][method],
                "--^",
                label=f"{method}, CPU",
                markersize=4,
            )
            speedup = [
                cpu / gpu
                for cpu, gpu in zip(
                    measurements["cpu"]["times"][method], measurements["gpu"]["times"][method]
                )
            ]
            right.semilogx(counts, speedup, "-o", label=method, markersize=5)

        left.set_xlabel("Number of function evaluations $N$")
        left.set_ylabel("Wall-clock time per integration [s]")
        left.set_title("GPU vs CPU runtime (separate processes)", fontsize=10)
        left.grid(True, which="both", alpha=0.3)
        left.legend(fontsize=8)

        right.axhline(1.0, color="k", ls=":", lw=1)
        right.set_xlabel("Number of function evaluations $N$")
        right.set_ylabel("CPU time / GPU time")
        right.set_title("GPU speedup (>1 means the GPU wins)", fontsize=10)
        right.grid(True, which="both", alpha=0.3)
        right.legend(fontsize=9)

        figure.suptitle(
            f"torchquad runtime scaling, 1D $\\sin x$ | {_hardware_label('float32')}",
            fontsize=11,
        )
        figure.tight_layout()
        _save(figure, measurements, "torchquad_runtime_cpu_vs_gpu", precision="float32")
    return measurements


_PLOTS = {
    "convergence": plot_convergence,
    "qmc": plot_qmc_vs_mc,
    "dimension": plot_dimension_scaling,
    "runtime": plot_runtime_cpu_vs_gpu,
}


def main():
    """Parse arguments and generate the requested figures."""
    parser = argparse.ArgumentParser(description="Generate README performance figures")
    parser.add_argument(
        "--plots",
        default="all",
        help=f"Comma-separated subset of {sorted(_PLOTS)}, or 'all'.",
    )
    parser.add_argument(
        "--worker-runtime",
        choices=("cpu", "gpu"),
        help="Internal: measure runtime for one device and print JSON. Not for direct use.",
    )
    arguments = parser.parse_args()

    if arguments.worker_runtime:
        _runtime_worker(arguments.worker_runtime)
        return

    # float64 is not optional here: float32 floors the error near 1e-7 and hides
    # the difference between the higher-order rules entirely.
    set_up_backend("torch", data_type="float64")
    print(f"Backend ready: {_hardware_label()}")

    selected = sorted(_PLOTS) if arguments.plots == "all" else arguments.plots.split(",")
    for name in selected:
        if name not in _PLOTS:
            raise SystemExit(f"Unknown plot {name!r}; expected one of {sorted(_PLOTS)}")
        print(f"\n=== {name} ===")
        _PLOTS[name]()


if __name__ == "__main__":
    sys.exit(main())

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(evals, fvals, ground_truth, labels, dpi=150):
    """Plots errors vs. function evaluations (fevals) and shows the convergence rate.

    Args:
        evals (list of np.array): Number of evaluations, for each method a np.array of ints.
        fvals (list of np.array): Function values for evals.
        ground_truth (np.array): Ground truth values.
        labels (list): Method names.
        dpi (int, optional): Plot dpi. Defaults to 150.
    """
    fig = plt.figure(dpi=dpi)
    n = 0
    for evals_item, f_item, label in zip(evals, fvals, labels):
        evals_item = np.array(evals_item)
        abs_err = np.abs(np.asarray(f_item) - np.asarray(ground_truth))
        abs_err_delta = np.mean(np.abs((abs_err[:-1]) / (abs_err[1:] + 1e-16)))
        label = label + "\nConvergence Rate: " + str.format("{:.2e}", abs_err_delta)
        plt.semilogy(evals_item, abs_err, label=label)

    plt.legend(fontsize=6)
    plt.xlabel("# of function evaluations")
    plt.ylabel("Absolute error")

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(evals, fvals, ground_truth, labels, dpi=150):
    """Plots errors vs fevals

    Args:
        evals (list of np.array): number of evaluations, for each method a np.array of fevals]
        fvals (list of np.array): function values for evals
        ground_truth (np.array): ground_truth values
        labels (list): method names
        dpi (int, optional): plot dpi. Defaults to 150.
    """
    fig = plt.figure(dpi=dpi)
    for evals_item, f_item, label in zip(evals, fvals, labels):
        abs_err = np.abs(np.asarray(f_item) - np.asarray(ground_truth))
        plt.semilogy(evals_item, abs_err, label=label)
    plt.legend()
    plt.xlabel("# of function evaluations")
    plt.ylabel("Absolute error")

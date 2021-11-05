import matplotlib.pyplot as plt


def plot_runtime(evals, runtime, labels, dpi=150, y_axis_name="Runtime [s]"):
    """Plots the runtime vs. function evaluations (fevals).

    Args:
        evals (list of np.array): Number of evaluations, for each method a np.array of fevals.
        runtime (list of np.array): Runtime for evals.
        labels (list): Method names.
        dpi (int, optional): Plot dpi. Defaults to 150.
        y_axis_name (str, optional): Name for y axis. Deafults to "Runtime [s]".
    """
    plt.figure(dpi=dpi)
    for evals_item, rt, label in zip(evals, runtime, labels):
        plt.semilogy(evals_item, rt, label=label)
    plt.legend(fontsize=6)
    plt.xlabel("Number of evaluations")
    plt.ylabel(y_axis_name)

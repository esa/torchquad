import matplotlib.pyplot as plt


def plot_runtime(evals, runtime, labels, dpi=150, y_axis_name="Runtime [s]"):
    """Plots runtime vs fevals

    Args:
        evals (list of np.array): number of evaluations, for each method a np.array of fevals]
        runtime (list of np.array): runtime for evals
        labels (list): method names
        dpi (int, optional): plot dpi. Defaults to 150.
        y_axis_name (str, optional): name for y axis. Deafult to ""Runtime [s]""
    """
    fig = plt.figure(dpi=dpi)
    for evals_item, rt, label in zip(evals, runtime, labels):
        plt.semilogy(evals_item, rt, label=label)
    plt.legend(fontsize=6)
    plt.xlabel("Number of evaluations")
    plt.ylabel(y_axis_name)

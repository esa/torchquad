import matplotlib.pyplot as plt
import numpy as np


def plot_grid(grid, dpi=100):
    """Plots the adaptive grid and corresponding function value.

    Args:
        grid (AdaptiveGrid): AdaptiveGrid of evaluated function
        dpi (int, optional): Plot dpi. Defaults to 100.
    """

    fig = plt.figure(dpi=dpi)
    points = None
    fvals = None
    for subdomain in grid.subdomains:
        if points is None:
            points = subdomain.points
            fvals = subdomain.fval
        else:    
            points = np.concatenate([points,subdomain.points])
            fvals = np.concatenate([fvals,subdomain.fval])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], fvals, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(points[:,0], points[:,1], fvals, c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    plt.show()

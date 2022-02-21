import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_adaptive_grid(grid, dpi=100):
    """Plots the 2D-adaptive grid and corresponding function value.

    Args:
        grid (AdaptiveGrid): AdaptiveGrid of evaluated function
        dpi (int, optional): Plot dpi. Defaults to 100.
    """

    if grid._dim != 2:
        raise ValueError('Plotting works only on 2D functions')

    fig = plt.figure(dpi=dpi,figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')  

    #Initialize colormap and patches for legend
    max_refinement_level = -1
    for subdomain in grid.subdomains:
        if subdomain.refinement_level > max_refinement_level:
            max_refinement_level = subdomain.refinement_level
    #max_refinement_level=grid._max_refinement_level
    cmap = plt.get_cmap('YlOrRd')
    colors = cmap(np.linspace(0, 1, max_refinement_level+1))
    patches = [mpatches.Patch(color=colors[rl], label=f'RL {rl+1}: not used') for rl in range(max_refinement_level)]


    for subdomain in grid.subdomains:
        points = np.concatenate([subdomain.points.cpu().numpy(),np.expand_dims(subdomain.fval.cpu().numpy(),axis=-1)],axis=-1)
        point_density = points.shape[0]/2000

        #reduces points in subdomain to ~2000 points if more
        if point_density > 1:
            #get bounds
            x,y = subdomain.integration_domain
            x = x.cpu().numpy()
            y = y.cpu().numpy()

            #get bounds indices
            x_lowerbound = np.where(points[:,0] == x[0])
            x_upperbound = np.where(points[:,0] == x[1])
            y_lowerbound = np.where(points[:,1] == y[0])
            y_upperbound = np.where(points[:,1] == y[1])

            #"downsample" inner grid, but keep boundary untouched (otherwise 3D surface will have gaps for high N)
            frame_index = np.concatenate([x_lowerbound,x_upperbound,y_lowerbound,y_upperbound],axis=-1)[0]
            points_without_frame = np.delete(points,frame_index,axis=0)
            points_without_frame_downsampled = points_without_frame[::int(point_density)]
            points = np.concatenate([points_without_frame_downsampled,points[frame_index]],axis=0)

        #run trisurf on remaining points and populate the plot legend
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2], color=colors[subdomain.refinement_level-1], edgecolors='none', alpha=1,linewidth=0,antialiased=False)
        patches[subdomain.refinement_level-1] = mpatches.Patch(color=colors[subdomain.refinement_level-1], label=f'RL {subdomain.refinement_level}: {int(point_density*2000)} points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    ax.legend(handles=patches,loc='lower right')
    plt.show()

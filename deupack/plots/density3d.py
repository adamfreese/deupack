# density3d.py
#
# Created by Adam Freese 2025.11.20, starting from code by Lorenz Sparrenberg:
# https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main 3D density plotting routine, adapted from Lorenz Sparrenberg:
# https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59
#
# modifications made by Adam Freese:
# - requires input coordinate arrays to be one-dimensional
# - allows divergent colormaps
# - allows 2D projections onto planes
#   - change in input arrays to accommodate this easily
#   - optional vmax parameter to allow consistent color scales across subplots
#   - optional gridlines on walls

def density3d(ax, x, y, z, values,
              decay=2.0, opacity=1.0, cmap=plt.cm.magma,
              projections=False, divergent=False, grid=True,
              vmax=-1, axi_x=-1, axi_y=-1, axi_z=-1,
              **kwargs):
    ''' Create a density plot for X, Y, Z coordinates and corresponding intensity values.
    Adapted from code by Lorentz Sparrenberg; see:
    https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59

    Input:
    -----------
    ax : plt.axis
        The axis object to plot the density plot on.
    x : np.array
        An array of X-coordinates; should be one-dimensional.
    y : np.array
        An array of Y-coordinates; should be one-dimensional.
    z : np.array
        An array of Z-coordinates; should be one-dimensional.
    values : np.ndarray
        An array of intensity values; should be three-dimensional.
    decay : float, optional
        The decay factor for the alpha values. Default is 2.0.
    opacity : float, optional
        The opacity value for the alpha values. Default is 1.0.
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors.
        Default is plt.cm.magma.
    projections: bool, optional
        Set to True to also plot 2D projections on the panes.
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
    grid: bool, optional
        If True, a grid will be drawn on the walls
    vmax: float, optional
        Maximum effective value for scaling the coloarmap.
        If <0 is passed, then the data will be used to determine the maximum value.
    axi_x : int, optional
        Index at which x axis is sliced for projection. Defaults to midpoint.
    axi_y : int, optional
        Index at which y axis is sliced for projection. Defaults to midpoint.
    axi_z : int, optional
        Index at which z axis is sliced for projection. Defaults to midpoint.
    **kwargs
        Additional keyword arguments to pass to the scatter function.

    Returns:
    --------
    None
    '''
    # Calculate RGB colors from intensities
    # Normalize the intensities between 0 and 1 and convert them to RGB colors using the chosen colormap
    if(vmax < 0):
        vmax = max(np.max(values), -np.min(values))
    if(divergent):
        # If we have negative values, we need to map the normed values onto [0,1]
        normed_values = (values + vmax) / (2*vmax)
        # We also need to make sure alpha is based on magnnitude
        alphas = (clip(abs(values) / abs(values.max()))) ** decay
    else:
        normed_values = values / vmax
        # Create alpha values for each data point based on its intensity and the specified decay factor
        alphas = (clip(values / values.max())) ** decay
    colors = cmap(clip(normed_values))
    alphas *= opacity
    colors[:, :, :, 3] = alphas  # add alpha values to RGB values
    # Flatten color array but keep last dimension
    colors_flattened = colors.reshape(-1, colors.shape[-1])
    # Create the 3D meshed grids
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    # Plot a 3D scatter with adjusted alphas
    ax.scatter(x_, y_, z_, c=colors_flattened, zorder=3, rasterized=True, **kwargs)
    # Project onto 2D planes if requested
    if(projections):
        if(axi_x < 0):
            axi_x = x.shape[0] // 2
        if(axi_y < 0):
            axi_y = y.shape[0] // 2
        if(axi_z < 0):
            axi_z = z.shape[0] // 2
        projection2d(ax, x, y, z, values, 'x', axi=axi_x, cmap=cmap, divergent=divergent, vmax=vmax)
        projection2d(ax, x, y, z, values, 'y', axi=axi_x, cmap=cmap, divergent=divergent, vmax=vmax)
        projection2d(ax, x, y, z, values, 'z', axi=axi_x, cmap=cmap, divergent=divergent, vmax=vmax)
        if(grid):
            add_gridlines(ax)
    return None

# The 2D projection routine called by the 3D density routine ~~~~~~~~~~~~~~~~~~~

def projection2d(ax, x, y, z, values, axis, axi=0,
                 cmap=plt.cm.magma, divergent=False, vmax=-1):
    ''' Calculates and plots a 2D projection of a 3D density onto one of its panes.

    Input:
    -----------
    ax : plt.axis
        The axis object to plot the density plot on.
    x : np.array
        An array of X-coordinates; should be one-dimensional.
    y : np.array
        An array of Y-coordinates; should be one-dimensional.
    z : np.array
        An array of Z-coordinates; should be one-dimensional.
    values : np.ndarray
        An array of intensity values; should be three-dimensional.
    axis : str
        Should be 'x', 'y' or 'z'; which axis is sliced.
    axi : int
        The index that the chosen axis should be sliced at.
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors.
        Default is plt.cm.magma.
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
    vmax: float, optional
        Maximum effective value for scaling the coloarmap in the 2D projections.
        If <0 is passed, then the data will be used to determine the maximum value.

    Returns:
    --------
    None
    '''
    # Stuff that varies between choice of axis
    if(axis=='x'):
        # Get 2D values by integrating out axis
        values2D = values[axi,:,:]
        # Make a meshgrid for non-integrated dimensions
        y_, z_ = np.meshgrid(y, z, indexing='ij')
        axmin, axmax = ax.get_xlim()
        # Fill an array for the integrated axis at the location its surface should be
        x_ = np.full_like(y_, axmin)
        # Identify which axis limits should be reset when we're done
        set_lim = ax.set_xlim
    elif(axis=='y'):
        values2D = values[:,axi,:]
        x_, z_ = np.meshgrid(x, z, indexing='ij')
        axmin, axmax = ax.get_ylim()
        y_ = np.full_like(x_, axmax)
        set_lim = ax.set_ylim
    elif(axis=='z'):
        values2D = values[:,:,axi]
        x_, y_ = np.meshgrid(x, y, indexing='ij')
        axmin, axmax = ax.get_zlim()
        z_ = np.full_like(x_, axmin)
        set_lim = ax.set_zlim
    else:
        raise ValueError("{} is not a valid axis; use 'x', 'y' or 'z'.".format(axis))
    # Colors for intensities
    if(vmax<0):
        if(divergent):
            vmax = max( np.max(values2D), -np.min(values2D) )
        else:
            vmax = np.max(values2D)
    if(divergent):
        colors = cmap(clip((values2D+vmax) / (2*vmax)))
    else:
        colors = cmap(clip(values2D / vmax))
    # Plot surfaces to emulate pcolormesh as best as possible
    ax.plot_surface(x_, y_, z_,
                    facecolors=colors, rstride=1, cstride=1,
                    shade=False, zorder=1, rasterized=True)
    # Reset the x, y or z limits in case they were moved
    set_lim((axmin, axmax))
    return

def add_gridlines(ax):
    # Get tick locations; they're where we'll make the lines
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    # Get the minimum and maximum x, y and z values for line extents
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()
    # Gridlines
    kwargs = {
            'linewidth' : 1,
            'zorder'    : 2,
            'color'     : 'gray',
            'alpha'     : 0.5
            }
    # Along x ticks (bottom & back walls)
    for x in xticks:
        if(x > xmin and x < xmax):
            ax.plot([x,x], [ymin, ymax], [zmin, zmin], '-', **kwargs)
            ax.plot([x,x], [ymax, ymax], [zmin, zmax], '-', **kwargs)
    # Along y ticks (bottom & left walls)
    for y in yticks:
        if(y > ymin and y < ymax):
            ax.plot([xmin,xmax], [y, y], [zmin, zmin], '-', **kwargs)
            ax.plot([xmin,xmin], [y, y], [zmin, zmax], '-', **kwargs)
    # Along z ticks (back & left walls)
    for z in zticks:
        if(z > zmin and z < zmax):
            ax.plot([xmin,xmax], [ymax, ymax], [z, z], '-', **kwargs)
            ax.plot([xmin,xmin], [ymin, ymax], [z, z], '-', **kwargs)
    # Reset axis limits
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.set_zlim((zmin,zmax))
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A multi-density plotting routine, tuned to deuteron needs

def multidensity3d(fig, x, y, z, nrows, ncols, *values_list,
                   labels=None, clabel=None, cmap=plt.cm.magma,
                   projections=False, divergent=False,
                   vmax=-1,
                   **kwargs):
    ''' Creates multiple 3D density plots in a single figure, along with a
    common colorbar for 2D projections at the bottom.
    Its parameters have been tuned to the needs of deuteron density plots.

    Input:
    -----------
    fig : plt.figure
        The figure object that all the subplots will go on.
    x : np.array
        An array of X-coordinates; should be one-dimensional.
    y : np.array
        An array of Y-coordinates; should be one-dimensional.
    z : np.array
        An array of Z-coordinates; should be one-dimensional.
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    values_list : list of np.ndarray
        Should have nrows*ncols entries, each of these entries being a
        three-dimensional array of intensity values. The list entries should
        be ordered from left to right, and then up to down (as if reading).
    labels : list of strings, optional
        If given, should have nrows*ncols entries, each of them being a string
        that labels a subplot (in the same order as values_list).
    clabel: str, optional
        Label for the colorbar.
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors.
        Default is plt.cm.magma.
    projections: bool, optional
        Set to True to also plot 2D projections on the panes.
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
    vmax: float, optional
        Give a positive number to manually set the colorbar scale.
    **kwargs
        Additional keyword arguments to pass to density3d.
        See docstring thereof for options.

    Returns:
    --------
    None
    '''
    N = 10 # how many times taller a subfigure should be than the colorbar
    # a gridspec is used to ensure the colorbar is smaller than other axes
    gs = fig.add_gridspec(nrows*N+1,ncols*N)
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            axes += [fig.add_subplot(gs[N*i:N*(i+1),N*j:N*(j+1)], projection='3d')]
    # Set up labels, pane colors, etc. for each subfigure
    for ax in axes:
        ax.set_xlabel('\n'+r'$x$ (fm)')
        ax.set_ylabel('\n'+r'$y$ (fm)')
        ax.set_zlabel('\n'+r'$z$ (fm)')
        ax.grid(False)
        ax.patch.set_alpha(0)
        ax.computed_zorder = False # TEST
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            if(divergent):
                axis.set_pane_color(cmap(0.5))
            else:
                axis.set_pane_color(cmap(0))
            axis.pane.set_edgecolor('gray')
    # Add another axis at the bottom for the colorbar
    axes += [fig.add_subplot(gs[nrows*N,:])]
    # Get the max value of any projections to use for scaling the colormap.
    if(vmax<0):
        vmax = np.max([ abs(p).max() for p in values_list ])
    # Add each of the 3D densities
    for n in range(len(values_list)):
        density3d(axes[n], x, y, z, values_list[n],
                  vmax=vmax, cmap=cmap,
                  projections=projections, divergent=divergent,
                  **kwargs)
    # Some labels
    if(labels is not None):
        bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='black', boxstyle='round,pad=0.5')
        for n in range(len(values_list)):
            axes[n].text2D(0.05,0.95, labels[n], fontsize=36, transform=axes[n].transAxes, bbox=bbox)
    # Colorbar!
    if(divergent):
        vmin = -vmax
    else:
        vmin = 0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=axes[-1],
            orientation='horizontal'
            )
    if(clabel is not None):
        cbar.set_label(clabel, size=36)
    # Some nice settings
    fig.tight_layout()
    fig.patch.set_alpha(0)
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clip(x):
    ''' Clips values of an array above 1 or below 0. '''
    x[x>1] = 1
    x[x<0] = 0
    return x

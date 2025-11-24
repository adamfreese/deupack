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

def density3d(ax, x, y, z, values,
              decay=2.0, opacity=1.0, cmap=plt.cm.magma,
              projections=False, divergent=False, vmax=0.0,
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
    vmax: float, optional
        Maximum effective value for scaling the coloarmap in the 2D projections.
        If 0 is passed, then the data will be used to determine the maximum value.
    **kwargs
        Additional keyword arguments to pass to the scatter function.

    Returns:
    --------
    None
    '''
    # Calculate RGB colors from intensities
    # Normalize the intensities between 0 and 1 and convert them to RGB colors using the chosen colormap
    if(divergent):
        # If we have negative values, we need to map the normed values onto [0,1]
        truemax = max(np.max(values), -np.min(values))
        normed_values = (values + truemax) / (2*truemax)
        # We also need to make sure alpha is based on magnnitude
        alphas = (abs(values) / truemax) ** decay
    else:
        normed_values = values / np.max(values)
        # Create alpha values for each data point based on its intensity and the specified decay factor
        alphas = (values / np.max(values)) ** decay
    colors = cmap(normed_values)
    alphas *= opacity
    colors[:, :, :, 3] = alphas  # add alpha values to RGB values
    # Flatten color array but keep last dimension
    colors_flattened = colors.reshape(-1, colors.shape[-1])
    # Create the 3D meshed grids
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    # Plot a 3D scatter with adjusted alphas
    ax.scatter(x_, y_, z_, c=colors_flattened, zorder=3, **kwargs)
    # Project onto 2D planes if requested
    if(projections):
        projection2d(ax, x, y, z, values, 'x', cmap=cmap, divergent=divergent, vmax=vmax)
        projection2d(ax, x, y, z, values, 'y', cmap=cmap, divergent=divergent, vmax=vmax)
        projection2d(ax, x, y, z, values, 'z', cmap=cmap, divergent=divergent, vmax=vmax)
    return None

# The 2D projection routine called by the 3D density routine ~~~~~~~~~~~~~~~~~~~

def projection2d(ax, x, y, z, values, axis,
                 cmap=plt.cm.magma, divergent=False, vmax=0.0):
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
        Should be 'x', 'y' or 'z'; which axis is integrated out
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors.
        Default is plt.cm.magma.
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
    vmax: float, optional
        Maximum effective value for scaling the coloarmap in the 2D projections.
        If 0 is passed, then the data will be used to determine the maximum value.

    Returns:
    --------
    None
    '''
    # Stuff that varies between choice of axis
    if(axis=='x'):
        # Get 2D values by integrating out axis
        values2D = np.trapz(values, x=x, axis=0)
        # Make a meshgrid for non-integreated dimensions
        y_, z_ = np.meshgrid(y, z, indexing='ij')
        axmin, axmax = ax.get_xlim()
        # Fill an array for the integrated axis at the location its surface should be
        x_ = np.full_like(y_, axmin)
        # Identify which axis limits should be reset when we're done
        set_lim = ax.set_xlim
    elif(axis=='y'):
        values2D = np.trapz(values, x=y, axis=1)
        x_, z_ = np.meshgrid(x, z, indexing='ij')
        axmin, axmax = ax.get_ylim()
        y_ = np.full_like(x_, axmax)
        set_lim = ax.set_ylim
    elif(axis=='z'):
        values2D = np.trapz(values, x=z, axis=2)
        x_, y_ = np.meshgrid(x, y, indexing='ij')
        axmin, axmax = ax.get_zlim()
        z_ = np.full_like(x_, axmin)
        set_lim = ax.set_zlim
    else:
        raise ValueError("{} is not a valid axis; use 'x', 'y' or 'z'.".format(axis))
    # Colors for intensities
    if(vmax==0.0):
        if(divergent):
            vmax = max( np.max(values2D), -np.min(values2D) )
        else:
            vmax = np.max(values2D)
    if(divergent):
        colors = cmap((values2D+vmax) / (2*vmax))
    else:
        colors = cmap(values2D / vmax)
    # Plot surfaces to emulate pcolormesh as best as possible
    ax.plot_surface(x_, y_, z_, facecolors=colors, rstride=1, cstride=1, shade=False, zorder=2)
    # Reset the x, y or z limits in case they were moved
    set_lim((axmin, axmax))
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A multi-density plotting routine, tuned to deuteron needs

def multidensity3d(fig, x, y, z, nrows, ncols, *values_list,
                   labels=None, clabel=None, cmap=plt.cm.magma, divergent=False,
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
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
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
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            if(divergent):
                axis.set_pane_color(cmap(0.5))
            else:
                axis.set_pane_color(cmap(0))
            axis.pane.set_edgecolor('gray')
    # Add another axis at the bottom for the colorbar
    axes += [fig.add_subplot(gs[nrows*N,:])]
    # Get the max value of any projections to use for scaling the colormap.
    # TODO: nicer code for getting vmax
    maxes = [
            [ abs(np.trapz(p, x=[x,y,z][axis], axis=axis)).max() for p in values_list ]
            for axis in [0,1,2]
            ]
    vmax = np.max(maxes)
    # Add each of the 3D densities
    for n in range(len(values_list)):
        density3d(axes[n], x, y, z, values_list[n],
                  vmax=vmax, cmap=cmap, divergent=divergent, **kwargs)
    # Some labels
    if(labels is not None):
        bbox = dict(facecolor='white', alpha=0.76, edgecolor='black', boxstyle='round,pad=0.5')
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


# densityplot3d.py
#
# Adapted from Lorenz Sparrenberg,
# https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59
#
# modifications made by Adam Freese:
# - allow 2D projections onto planes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def densityplot3d(ax: plt.axis,
                  x: np.ndarray,
                  y: np.ndarray,
                  z: np.ndarray,
                  values: np.ndarray,
                  decay: float = 2.0,
                  opacity: float = 1.0,
                  cmap: mpl.colors.LinearSegmentedColormap = plt.cm.jet,
                  projections: bool = False,
                  divergent: bool = False,
                  **kwargs) -> None:
    """
    Create a density plot for X, Y, Z coordinates and corresponding intensity values.

    Parameters:
    -----------
    ax : plt.axis
        The axis object to plot the density plot on.
    x : np.array
        An array of X-coordinates; should be one-dimensional.
    y : np.array
        An array of Y-coordinates. should be one-dimensional.
    z : np.array
        An array of Z-coordinates. should be one-dimensional.
    values : np.ndarray
        An array of intensity values; should be three-dimensional.
    decay : float, optional
        The decay factor for the alpha values. Default is 2.0.
    opacity : float, optional
        The opacity value for the alpha values. Default is 1.0.
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors. Default is plt.cm.jet.
    projections: bool, optional
        Set to True to also plot 2D projections on the panes.
    divergent: bool, optional
        Set to True if the values can go negative and a divergent colormap is used.
    **kwargs
        Additional keyword arguments to pass to the scatter function.

    Returns:
    --------
    None
    """
    # Changes by AF, 2025.11.20:
    # - x, y and z are expected to be 1D arrays; meshing is done internally
    # - projections boolean can be used to plot 2D projections too

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

    # New projection stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(projections):
        # Obtain proejections by integrating out one dimension
        v_xy = np.trapz(values, x=z, axis=2)
        v_yz = np.trapz(values, x=x, axis=0)
        v_zx = np.trapz(values, x=y, axis=1)

        # Colors for intensities
        if(divergent):
            # For divergent colormap, need to shift center to 0.5
            vmax = max(
                    np.max(v_xy), np.max(v_yz), np.max(v_zx)
                    -np.min(v_xy), -np.min(v_yz), -np.min(v_zx)
                    )
            c_xy = cmap((v_xy+vmax) / (2*vmax))
            c_yz = cmap((v_yz+vmax) / (2*vmax))
            c_zx = cmap((v_zx+vmax) / (2*vmax))
        else:
            vmax = max(np.max(v_xy), np.max(v_yz), np.max(v_zx))
            c_xy = cmap(v_xy / vmax)
            c_yz = cmap(v_yz / vmax)
            c_zx = cmap(v_zx / vmax)

        # Flatten out the color array, keeping last dimension
        c_xy_fl = c_xy.reshape(-1, c_xy.shape[-1])
        c_yz_fl = c_yz.reshape(-1, c_yz.shape[-1])
        c_zx_fl = c_zx.reshape(-1, c_zx.shape[-1])

        # Create the 2D meshed grids needed for the projections
        x_xy, y_xy = np.meshgrid(x, y, indexing='ij')
        y_yz, z_yz = np.meshgrid(y, z, indexing='ij')
        x_zx, z_zx = np.meshgrid(x, z, indexing='ij')

        # Get the x, y and z limits to make sure we project on the panes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        zmin, zmax = ax.get_zlim()

        # Do the projections as scatter plots, because pcolormesh
        # doesn't seem to be supported for this. =(
        ax.scatter(x_xy, y_xy, zs=zmin, zdir='z', c=c_xy_fl, zorder=2, **kwargs)
        ax.scatter(y_yz, z_yz, zs=xmin, zdir='x', c=c_yz_fl, zorder=2, **kwargs)
        ax.scatter(x_zx, z_zx, zs=ymax, zdir='y', c=c_zx_fl, zorder=2, **kwargs)

        # Reset the x, y and z limits in case they were moved
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_zlim((zmin, zmax))
    # End new projection stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return None

# densityplot3d.py
# Adapted from Lorenz Sparrenberg,
# https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59

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
                  **kwargs) -> None:
    """
    Create a density plot for X, Y, Z coordinates and corresponding intensity values.

    Parameters:
    -----------
    ax : plt.axis
        The axis object to plot the density plot on.
    x : np.ndarray
        An array of X-coordinates.
    y : np.ndarray
        An array of Y-coordinates.
    z : np.ndarray
        An array of Z-coordinates.
    values : np.ndarray
        An array of intensity values.
    decay : float, optional
        The decay factor for the alpha values. Default is 2.0.
    opacity : float, optional
        The opacity value for the alpha values. Default is 1.0.
    cmap : mpl.colors.LinearSegmentedColormap, optional
        The colormap used for mapping intensity values to RGB colors. Default is plt.cm.jet.
    **kwargs
        Additional keyword arguments to pass to the scatter function.

    Returns:
    --------
    None
    """

    # Calculate RGB colors from intensities
    # Normalize the intensities between 0 and 1 and convert them to RGB colors using the chosen colormap
    normed_values = values / np.max(values)
    colors = cmap(normed_values)
    
    # Create alpha values for each data point based on its intensity and the specified decay factor
    alphas = (values / np.max(values)) ** decay
    alphas *= opacity
    
    colors[:, :, :, 3] = alphas  # add alpha values to RGB values
    
    # Flatten color array but keep last dimension
    colors_flattened = colors.reshape(-1, colors.shape[-1])  

    # Plot a 3D scatter with adjusted alphas
    ax.scatter(x, y, z, c=colors_flattened, **kwargs)
    
    return None

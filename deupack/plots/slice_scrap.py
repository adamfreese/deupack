import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import mff
from ..density import *
from ..mff.nucleon.chooser import choose_nff

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2D density plots of slices

def plot_mass_slices():
    ''' Plot four slices of the mass density.
    These include two polarizations (mj=0 and mj=1)
    and two zero axes (x and z).
    '''
    # Set things up to make the four panels
    D = Density()
    nrows,ncols=1,4
    fig = plt.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1, aspect='equal')
    ax2 = plt.subplot(nrows,ncols,2, aspect='equal')
    ax3 = plt.subplot(nrows,ncols,3, aspect='equal')
    ax4 = plt.subplot(nrows,ncols,4, aspect='equal')
    # Plot the masses
    _ = plot_one_mass_slice(D, ax1, mj=1, zero_axis='z', ylabel=True)
    _ = plot_one_mass_slice(D, ax2, mj=0, zero_axis='z', ylabel=False)
    _ = plot_one_mass_slice(D, ax3, mj=1, zero_axis='x', ylabel=True)
    c = plot_one_mass_slice(D, ax4, mj=0, zero_axis='x', ylabel=False)
    # Save
    fig.patch.set_alpha(0)
    fig.savefig('mass.pdf')
    return

def plot_one_mass_slice(D, ax, zero_axis='z', mj=0, ylabel=False):
    ''' Plot one of the four mass slices. '''
    # Set up the arrays for the plotted axes
    b = np.linspace(-2.12, 2.12, 400)
    if(zero_axis=='z'):
        x,y,z = b,b,0
        xlabel = r'$x$ (fm)'
        ylabel = r'$y$ (fm)'
    elif(zero_axis=='x'):
        x,y,z = 0,b,b
        xlabel = r'$y$ (fm)'
        ylabel = r'$z$ (fm)'
    elif(zero_axis=='y'):
        x,y,z = b,0,b
        xlabel = r'$x$ (fm)'
        ylabel = r'$z$ (fm)'
    else:
        raise ValueError("Invalid value for zero_azis: {}.".format(zero_axis))
    # Obtain the density. Squeeze everything down to 2-dimensional grids
    MU = np.squeeze( D.mass_3D_U(x,y,z) )
    MT = np.squeeze( D.mass_3D_T(x,y,z) )
    if(mj==0):
        M = MU + 2/3*MT
    elif(mj==1 or mj==-1):
        M = MU - 1/3*MT
    else:
        raise ValueError("mj={:d} is not a valid spin.".format(mj))
    # Plot the slice
    vmax = MU.max() + abs(MT).max()
    c = ax.pcolormesh(b, b, M.T, vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    # Label and leave
    ax.annotate(
            r'$m_j={:d},\, {}=0$'.format(mj,zero_axis),
            (0.025,0.025), xycoords='axes fraction',
            color='white'
            )
    ax.set_xlabel(xlabel)
    if(ylabel):
        ax.set_ylabel(ylabel)
    else:
        ax.get_yaxis().set_ticks([])
    return c

def plot_stress_slice(zero_axis='z', mj=0):
    ''' Plot a 2D slice of the 3D stresses, with one of the three coordinate
    axes set to zero.
    '''
    # Set up the arrays for the plotted axes
    b = np.linspace(-2.12, 2.12, 200)
    if(zero_axis=='z'):
        x,y,z = b,b,0
        xlabel = r'$x$ (fm)'
        ylabel = r'$y$ (fm)'
    elif(zero_axis=='x'):
        x,y,z = 0,b,b
        xlabel = r'$y$ (fm)'
        ylabel = r'$z$ (fm)'
    elif(zero_axis=='y'):
        x,y,z = b,0,b
        xlabel = r'$x$ (fm)'
        ylabel = r'$z$ (fm)'
    else:
        raise ValueError("Invalid value for zero_azis: {}.".format(zero_axis))
    # Obtain the densities. Squeeze everything down to 2-dimensional grids
    D = Density()
    TijU   = np.squeeze( D.stress_3D_U(x,y,z) )
    TijT   = np.squeeze( D.stress_3D_T(x,y,z) )
    rhat   = np.squeeze( make_rhat(x,y,z)     )
    phihat = np.squeeze( make_phihat(x,y,z)   )
    zhat   = np.squeeze( make_zhat(x,y,z)     )
    prU = np.einsum('xyij,xyi,xyj->xy', TijU, rhat, rhat)
    ptU = np.einsum('xyij,xyi,xyj->xy', TijU, phihat, phihat)
    pzU = np.einsum('xyij,xyi,xyj->xy', TijU, zhat, zhat)
    prT = np.einsum('xyij,xyi,xyj->xy', TijT, rhat, rhat)
    ptT = np.einsum('xyij,xyi,xyj->xy', TijT, phihat, phihat)
    pzT = np.einsum('xyij,xyi,xyj->xy', TijT, zhat, zhat)
    if(mj==0):
        pr = prU + 2/3*prT
        pt = ptU + 2/3*ptT
        pz = pzU + 2/3*pzT
    elif(mj==1 or mj==-1):
        pr = prU - 1/3*prT
        pt = ptU - 1/3*ptT
        pz = pzU - 1/3*pzT
    else:
        raise ValueError("mj={:d} is not a valid spin.".format(mj))
    vmax = max( abs(pr).max(), abs(pt).max(), abs(pz).max() )
    # Create the plot canvas
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1, aspect='equal')
    ax2 = plt.subplot(nrows,ncols,2, aspect='equal')
    ax3 = plt.subplot(nrows,ncols,3, aspect='equal')
    # Plot the stresses
    c1 = ax1.pcolormesh(b, b, pr.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    c2 = ax2.pcolormesh(b, b, pt.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    c3 = ax3.pcolormesh(b, b, pz.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    # Color bar
    cbar = fig.colorbar(c3)
    cbar.set_label(r'Stress (GeV/fm$^3$)')
    # Labels
    ax1.set_title('Radial stress')
    ax2.set_title('Azimuthal stress')
    ax3.set_title('$z$-direction stress')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(xlabel)
        ax.annotate(
                r'$m_j={:d},\, {}=0$'.format(mj,zero_axis),
                (0.025,0.025), xycoords='axes fraction',
                color='white'
                )
    ax1.set_ylabel(ylabel)
    for ax in [ax2,ax3]:
        ax.get_yaxis().set_ticks([])
    # Save
    fig.patch.set_alpha(0)
    fig.savefig('stress_mj{:d}_{}0.pdf'.format(mj,zero_axis))
    return


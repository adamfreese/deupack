# AlanDevPlots.py
#
# Created 2025.12.02
#
# Routines for the plots included in our first deuteron stress paper.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmasher as cmr

from deupack import emtff
from deupack.density import Density
from deupack.densityLF import DensityLF
from deupack.plots.density3d import multidensity3d

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One-shot routine to generate all plots used in the paper
# NOTE: will be slow, especially on the first run

def make_dev_plots():
    ''' 
    A one-shot routine to make all the plots for quark gluon separated densities of deuteron and LF densities
    '''
    pressure()
    principal_axes()   
    principal_axesLF()
    forces()           
    forcesLF()
    return






def pressure():
    # Fixed parameters for the visualization
    nff='bag'; wf='av18'; nb=101; bmax=2
    # Get the pressures
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    pr0 = D.isoradial_pressure(pol=0)
    pθ0 = D.isopolar_pressure( pol=0)
    pφ0 = D.azimuthal_pressure(pol=0)
    pr1 = D.isoradial_pressure(pol=1)
    pθ1 = D.isopolar_pressure( pol=1)
    pφ1 = D.azimuthal_pressure(pol=1)
    # Make some labels
    labels = [
            r'$m_j=0$, isoradial pressure',
            r'$m_j=0$, isopolar pressure',
            r'$m_j=0$, azimuthal pressure',
            r'$m_j=\pm1$, isoradial pressure',
            r'$m_j=\pm1$, isopolar pressure',
            r'$m_j=\pm1$, azimuthal pressure'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    # Use the custom multi-3D plotter
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr0, pθ0, pφ0, pr1, pθ1, pφ1,
                   labels=labels,
                   clabel=r'Pressure (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('pressure3DGluons.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Two-dimensional density/quiver/streamline plots

def principal_axes():
    # Parameters for this visualization (fixed)
    bmax = 1.6; nff='bag'; wf='av18'; nbq = 21; nbh = 101
    # Density objects for quiver (small) and heat map (large)
    Dq = Density(nff=nff, wf=wf, bmax=bmax, nb=nbq)
    Dh = Density(nff=nff, wf=wf, bmax=bmax, nb=nbh)
    # Prepare figure
    nrows,ncols=2,2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    ax0p = axes[0,0]
    ax0m = axes[0,1]
    ax1p = axes[1,0]
    ax1m = axes[1,1]
    for ax in [ax0p, ax0m, ax1p, ax1m]:
        ax.set_aspect('equal')
    vmax = np.max([
        abs(Dh.isoradial_pressure(pol=0)).max(),
        abs(Dh.isoradial_pressure(pol=1)).max(),
        abs(Dh.isopolar_pressure( pol=0)).max(),
        abs(Dh.isopolar_pressure( pol=1)).max()
        ])
    # Call the panel code four times
    _ = _eigenvector_panel(ax0p, Dq, Dh, '+', 0, vmax, r'Isoradial,  $m_j=0$')
    _ = _eigenvector_panel(ax0m, Dq, Dh, '-', 0, vmax, r'Isopolar, $m_j=0$')
    _ = _eigenvector_panel(ax1p, Dq, Dh, '+', 1, vmax, r'Isoradial,  $m_j=\pm1$')
    _ = _eigenvector_panel(ax1m, Dq, Dh, '-', 1, vmax, r'Isopolar, $m_j=\pm1$')
    # Remove x axes from top two panels for economic use of space
    for ax in [ax0p, ax0m]:
        ax.get_xaxis().set_visible(False)
    # Remove y axes from right two panels for the same reason
    for ax in [ax0m, ax1m]:
        ax.get_yaxis().set_visible(False)
    # Make the colorbar
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.fusion_r),
            ax = axes[:, 1],
            orientation='vertical',
            )
    cbar.set_label(r'Pressure (GeV/fm$^3$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('principal_axesGluons.pdf', bbox_inches="tight")
    return



def principal_axesLF():
    # Parameters for this visualization (fixed)
    bmax = 0.7; nff='bag'; nbq=21 ;nbh = 101


    SpinZ = (0.,0.,1.)
    SpinY = (0.,1.,0.)
    SpinX = (1.,0.,0.)



    # Density objects for quiver (small) and heat map (large)
    DqZ = DensityLF(nff=nff,  bmax=bmax, nb=nbq,SpinV=SpinZ)
    DhZ = DensityLF(nff=nff,  bmax=bmax, nb=nbh,SpinV=SpinZ)
    DqY = DensityLF(nff=nff,  bmax=bmax, nb=nbq,SpinV=SpinY)
    DhY = DensityLF(nff=nff,  bmax=bmax, nb=nbh,SpinV=SpinY)
    DqX = DensityLF(nff=nff,  bmax=bmax, nb=nbq,SpinV=SpinX)
    DhX = DensityLF(nff=nff,  bmax=bmax, nb=nbh,SpinV=SpinX)
    # Prepare figure
    nrows,ncols=2,3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    axZp = axes[0,0]
    axZm = axes[1,0]
    axYp = axes[0,1]
    axYm = axes[1,1]
    axXp = axes[0,2]
    axXm = axes[1,2]
    for ax in [axZp, axZm, axYp, axYm, axXp, axXm]:
        ax.set_aspect('equal')
    vmax = np.max([
        abs(DhZ.isoradial_pressure()).max(),
        abs(DhX.isoradial_pressure()).max(),
        abs(DhY.isoradial_pressure()).max(),
        abs(DhZ.isoazimuthal_pressure()).max(),
        abs(DhY.isoazimuthal_pressure()).max(),
        abs(DhX.isoazimuthal_pressure()).max(),
        ])
    # Call the panel code four times
    _ = _eigenvector_panel_LF(axZp, DqZ, DhZ, '+', vmax, r'Isoradial,  z')
    _ = _eigenvector_panel_LF(axZm, DqZ, DhZ, '-', vmax, r'Isoazimuthal, z')
    _ = _eigenvector_panel_LF(axYp, DqY, DhY, '+', vmax, r'Isoradial,  y')
    _ = _eigenvector_panel_LF(axYm, DqY, DhY, '-', vmax, r'Isoazimuthal, y')
    _ = _eigenvector_panel_LF(axXp, DqX, DhX, '+', vmax, r'Isoradial,  x')
    _ = _eigenvector_panel_LF(axXm, DqX, DhX, '-', vmax, r'Isoazimuthal, x')
    # Remove x axes from top three panels for economic use of space
    for ax in [axZp, axYp,axXp]:
        ax.get_xaxis().set_visible(False)
    # Remove y axes from middle and right panels for the same reason
    for ax in [axXp, axXm,axYp, axYm]:
        ax.get_yaxis().set_visible(False)
    # Make the colorbar
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.fusion_r),
            ax = axes[:, 2],
            orientation='vertical',
            )
    cbar.set_label(r'Pressure (GeV/fm$^2$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('principal_axesProtonGluons.pdf', bbox_inches="tight")
    return




def forces():
    # Fixed parameters
    nff = 'bag'; wf = 'av18'; bmax = 1.4; nb = 101
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    # Get vmax
    vmax = np.max([
        abs(D.radial_force(pol=0)).max(),
        abs(D.polar_force( pol=0)).max(),
        abs(D.radial_force(pol=1)).max(),
        abs(D.polar_force( pol=1)).max()
        ])
    # Prepare figure
    nrows,ncols=1,2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    ax0 = axes[0]
    ax1 = axes[1]
    norm = mpl.colors.LogNorm(vmin=1e-3*vmax, vmax=vmax)
    _ = _force_panel_stream(ax0, D, pol=0, norm=norm, label=r'$m_j=0$')
    _ = _force_panel_stream(ax1, D, pol=1, norm=norm, label=r'$m_j=\pm 1$')
    # Remove y axes from right panel to save space
    ax1.get_yaxis().set_visible(False)
    # Make the colorbar
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.voltage_r),
            ax = axes[1],
            orientation='vertical',
            )
    cbar.set_label(r'Force density (GeV/fm$^4$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('forcesGluons.pdf', bbox_inches="tight")
    return


def forcesLF():
    # Fixed parameters
    nff = 'bag'; bmax = 1.5; nb = 101
    SpinZ = (0.,0.,1.)
    D = DensityLF(nff=nff, bmax=bmax, nb=nb,SpinV=SpinZ)
    SpinY = (0.,1.,0.)
    D1 = DensityLF(nff=nff, bmax=bmax, nb=nb,SpinV=SpinY)
    SpinX = (1.,0.,0.)
    D2 = DensityLF(nff=nff, bmax=bmax, nb=nb,SpinV=SpinX)
    # Get vmax
    vmax = np.max([
        abs(D.radial_force()).max(),
        abs(D1.radial_force()).max(),
        abs(D2.radial_force()).max(),
        abs(D.azimuthal_force()).max(),
        abs(D1.azimuthal_force()).max(),
        abs(D2.azimuthal_force()).max()
        ])
    # Prepare figure
    nrows,ncols=1,3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    norm = mpl.colors.LogNorm(vmin=1e-3*vmax, vmax=vmax)
    # norm = mpl.colors.Normalize(vmin=1e-3*vmax, vmax=vmax)
    _ = _force_panel_streamLF(ax0, D, norm=norm, label=r'z')
    _ = _force_panel_streamLF(ax1, D1, norm=norm, label=r'y')
    _ = _force_panel_streamLF(ax2, D2, norm=norm, label=r'x')
    # Remove y axes from right panel to save space
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    # Make the colorbar
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.voltage_r),
            ax = axes[2],
            orientation='vertical',
            )
    cbar.set_label(r'Force density (GeV/fm$^3$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('forcesProtonGluons.pdf', bbox_inches="tight")
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities for 2D plots

def _doublequiver(ax, x, y, vx, vy):
    quiver_kwargs = {
            'color' : 'white',
            'alpha' : 0.7,
            'angles' : 'xy',
            'scale_units' : 'xy',
            'pivot' : 'tail',
            'scale' : x.shape[0]/x.max(),
            'width' : 0.003
            }
    q1 = ax.quiver(x, y,  vx.T,  vy.T, **quiver_kwargs)
    q2 = ax.quiver(x, y, -vx.T, -vy.T, **quiver_kwargs)
    for q in [q1, q2]:
        q.set_path_effects([pe.Stroke(linewidth=3, foreground='black', alpha=0.3), pe.Normal()])
    return

def _eigenvector_panel(ax, Dq, Dh, mode, pol, vmax, label):
    if(mode=='+'):
        X, Y, Z = Dq.e_plus(pol=pol)
        P = Dh.isoradial_pressure(pol=pol)
    elif(mode=='-'):
        X, Y, Z = Dq.e_minus(pol=pol)
        P = Dh.isopolar_pressure(pol=pol)
    else:
        raise ValueError("Invalid mode: {}; expected + or -.".format(mode))
    # Slice at y=0
    bq = Dq.x
    nbq = bq.shape[0]
    x = X[:,nbq//2,:]
    z = Z[:,nbq//2,:]
    bh = Dh.x
    nbh = bh.shape[0]
    p = P[:,nbh//2,:]
    # Heat map first
    c = ax.pcolormesh(bh, bh, p.T, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    # Quiver plot next
    _doublequiver(ax, bq, bq, x, z)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$x$ (fm)')
    ax.set_ylabel(r'$z$ (fm)')
    return c


def _eigenvector_panel_LF(ax, Dq, Dh, mode, vmax, label):
    if(mode=='+'):
        X, Y= Dq.e_plus()
        P = Dh.isoradial_pressure()
    elif(mode=='-'):
        X, Y = Dq.e_minus()
        P = Dh.isoazimuthal_pressure()
    else:
        raise ValueError("Invalid mode: {}; expected + or -.".format(mode))
    bq = Dq.x
    x = Y #axis flipping
    y = X
    
    bh = Dh.x
    # Heat map first
    c = ax.pcolormesh(bh, bh, P, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    # Quiver plot next
    _doublequiver(ax, bq, bq, x.T, y.T)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$y$ (fm)')
    ax.set_ylabel(r'$x$ (fm)')
    return c

def _force_panel_stream(ax, D, pol, norm, label):
    # First, calculate the quivers for force directions
    b = D.x
    nb = b.shape[0]
    # Get force vectors ... sliced down to y=0
    fr = D.radial_force(pol=pol)[:,nb//2,:]
    fθ = D.polar_force( pol=pol)[:,nb//2,:]
    # Pull out the angular dependence
    theta = D.theta[:,nb//2,:]
    phi = D.phi[:,nb//2,:]
    # Force in Cartesian coordinates, and its magnitude
    fz = (fr*np.cos(theta) - fθ*np.sin(theta))
    fx = (fr*np.sin(theta) + fθ*np.cos(theta)) * np.cos(phi)
    f = np.sqrt(fx**2 + fz**2)
    # Next, get the fine-grained force magnitude for the heat map
    # Plot the heat map
    c = ax.pcolormesh(b, b, f.T, norm=norm, cmap=cmr.voltage_r, shading='gouraud')
    # Plot the streamlines
    s = ax.streamplot(b, b, fx.T, fz.T,
                      color='white',
                      arrowsize=1.7, arrowstyle='-|>',
                      broken_streamlines=True,
                      density=1.5
                      )
    # Tune the alphas for better visibility
    s.lines.set_alpha(0.53)
    for x in ax.get_children():
        if type(x)==mpl.patches.FancyArrowPatch:
            x.set_alpha(0.69)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$x$ (fm)')
    ax.set_ylabel(r'$z$ (fm)')
    return c





def _force_panel_streamLF(ax, D, norm, label):
    # First, calculate the quivers for force directions
    b = D.x
    # nb = b.shape[0]

    fr = D.radial_force()
    fphi = D.azimuthal_force()
    # Pull out the angular dependence
    phi = D.phi
    # Force in Cartesian coordinates, and its magnitude
    fx = fr*np.cos(phi) -fphi*np.sin(phi)
    fy = fr*np.sin(phi)+fphi*np.cos(phi)
    f = np.sqrt(fx**2 + fy**2)
    # Next, get the fine-grained force magnitude for the heat map
    # Plot the heat map
    c = ax.pcolormesh(b, b, f, norm=norm, cmap=cmr.voltage_r, shading='gouraud')
    # Plot the streamlines
    s = ax.streamplot(b, b, fy, fx,
                      color='white',
                      arrowsize=1.7, arrowstyle='-|>',
                      broken_streamlines=True,
                      density=1.9
                      )
    # Tune the alphas for better visibility
    s.lines.set_alpha(0.53)
    for x in ax.get_children():
        if type(x)==mpl.patches.FancyArrowPatch:
            x.set_alpha(0.69)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$y$ (fm)')
    ax.set_ylabel(r'$x$ (fm)')
    return c

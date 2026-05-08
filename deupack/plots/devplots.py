# paperplots.py
#
# Created 2026.03.12 by Adam Freese, for some personal purposes.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmasher as cmr

from .. import emtff
from ..density import NucleonDensity as Density
from .density3d import multidensity3d

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pressure():
    # Fixed parameters for the visualization
    nff='mit'; nb=151; bmax=0.84
    # Get the pressures
    D = Density(nff=nff, nb=nb, bmax=bmax)
    pr = D.radial_pressure(pol=0)
    pt = D.polar_pressure( pol=0)
    # Make some labels
    labels = [
            r'radial pressure',
            r'tangential pressure'
            ]
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    # Use the custom multi-3D plotter
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr, pt,
                   labels=labels,
                   clabel=r'Pressure (GeV/fm$^3$)',
                   decay=1, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('nucleon_pressure.pdf')
    return

def pressure_separated():
    # Parameters for this visualization (fixed)
    bmax = 0.84; nbq = 21; nbh = 151
    # Density objects for quiver (small) and heat map (large)
    Dq = Density(nff='mit', bmax=bmax, nb=nbq)
    Dh = Density(nff='mit', bmax=bmax, nb=nbh)
    Dqq = Density(nff='mitq', bmax=bmax, nb=nbq)
    Dhq = Density(nff='mitq', bmax=bmax, nb=nbh)
    Dqg = Density(nff='mitg', bmax=bmax, nb=nbq)
    Dhg = Density(nff='mitg', bmax=bmax, nb=nbh)
    # Prepare figure
    nrows,ncols=2,3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    axtr = axes[0,0]
    axtt = axes[1,0]
    axqr = axes[0,1]
    axqt = axes[1,1]
    axgr = axes[0,2]
    axgt = axes[1,2]
    for ax in [axtr, axtt, axqr, axqt, axgr, axgt]:
        ax.set_aspect('equal')
    vmax = np.max([
        abs(Dh.radial_pressure(pol=0)).max(),
        abs(Dh.polar_pressure( pol=0)).max(),
        abs(Dhq.radial_pressure(pol=0)).max(),
        abs(Dhq.polar_pressure( pol=0)).max(),
        abs(Dhg.radial_pressure(pol=0)).max(),
        abs(Dhg.polar_pressure( pol=0)).max()
        ]) / 2
    # Call the panel code four times
    _ = _pressure_panel(axtr, Dq,  Dh,  '+', vmax, r'Radial (total)')
    _ = _pressure_panel(axtt, Dq,  Dh,  '-', vmax, r'Tangential (total)')
    _ = _pressure_panel(axqr, Dqq, Dhq, '+', vmax, r'Radial (quark)')
    _ = _pressure_panel(axqt, Dqq, Dhq, '-', vmax, r'Tangential (quark)')
    _ = _pressure_panel(axgr, Dqg, Dhg, '+', vmax, r'Radial (gluon)')
    _ = _pressure_panel(axgt, Dqg, Dhg, '-', vmax, r'Tangential (gluon)')
    # Remove x axes from top two panels for economic use of space
    for ax in [axqr, axqt, axgr, axgt]:
        ax.get_yaxis().set_visible(False)
    # Remove y axes from right two panels for the same reason
    for ax in [axtr, axqr, axgr]:
        ax.get_xaxis().set_visible(False)
    # Make the colorbar
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.fusion_r),
            ax = axes[:, 2],
            orientation='vertical',
            )
    cbar.set_label(r'Pressure (GeV/fm$^3$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('pressure_separated.pdf', bbox_inches="tight")
    return
    ## # Fixed parameters for the visualization
    ## nb=151; bmax=0.84
    ## # Get the pressures
    ## D = Density(nff='mit', nb=nb, bmax=bmax)
    ## Dq = Density(nff='mitq', nb=nb, bmax=bmax)
    ## Dg = Density(nff='mitg', nb=nb, bmax=bmax)
    ## pr = D.radial_pressure(pol=0)
    ## pt = D.polar_pressure( pol=0)
    ## prq = Dq.radial_pressure(pol=0)
    ## ptq = Dq.polar_pressure( pol=0)
    ## prg = Dg.radial_pressure(pol=0)
    ## ptg = Dg.polar_pressure( pol=0)
    ## # Make some labels
    ## labels = [
    ##         r'total radial pressure',
    ##         r'quark radial pressure',
    ##         r'gluon radial pressure',
    ##         r'total tangential pressure',
    ##         r'quark tangential pressure',
    ##         r'gluon tangential pressure'
    ##         ]
    ## # Prepare figure
    ## nrows,ncols=2,3
    ## fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    ## # Use the custom multi-3D plotter
    ## multidensity3d(fig, D.x, D.x, D.x,
    ##                nrows, ncols,
    ##                pr, prq, prg, pt, ptq, ptg,
    ##                labels=labels,
    ##                clabel=r'Pressure (GeV/fm$^3$)',
    ##                decay=1, opacity=0.69, cmap=cmr.fusion_r,
    ##                projections=True, divergent=True, s=1)
    ## fig.savefig('nucleon_pressure_separated.pdf')
    ## return

def force():
    # Fixed parameters
    nff = 'mitq'; bmax = 0.84; nb = 150
    D = Density(nff=nff, bmax=bmax, nb=nb)
    # Get vmax
    vmax = abs(D.radial_force(pol=0)).max()
    # Prepare figure
    nrows,ncols=1,1
    fig, ax0 = plt.subplots(nrows, ncols, figsize=(ncols*9.4,nrows*7.11), layout='constrained')
    norm = mpl.colors.LogNorm(vmin=1e-3*vmax, vmax=vmax)
    _ = _force_panel_stream(ax0, D, pol=0, norm=norm, label=r'Force density ($y=0$)')
    # Make the colorbar
    cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmr.voltage_r),
            ax = ax0,
            orientation='vertical',
            )
    cbar.set_label(r'Force density (GeV/fm$^4$)', size=36)
    fig.patch.set_alpha(0)
    fig.savefig('nucleon_force.pdf', bbox_inches="tight")
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

def _pressure_panel(ax, Dq, Dh, mode, vmax, label):
    if(mode=='+'):
        X, Y, Z = Dq.e_plus(pol=0)
        P = Dh.radial_pressure(pol=0)
    elif(mode=='-'):
        X, Y, Z = Dq.e_minus(pol=0)
        P = Dh.polar_pressure(pol=0)
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
    #s = ax.streamplot(b, b, fx.T, fz.T,
    #                  color='white',
    #                  arrowsize=1.7, arrowstyle='-|>',
    #                  broken_streamlines=False,
    #                  density=0.5
    #                  )
    ## Tune the alphas for better visibility
    #s.lines.set_alpha(0.11)
    #for x in ax.get_children():
    #    if type(x)==mpl.patches.FancyArrowPatch:
    #        x.set_alpha(0.22)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$x$ (fm)')
    ax.set_ylabel(r'$z$ (fm)')
    return

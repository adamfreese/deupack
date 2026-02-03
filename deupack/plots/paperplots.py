# paperplots.py
#
# Created 2025.12.02
#
# Routines for the plots included in our first deuteron stress paper.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmasher as cmr

from .. import emtff
from ..density import Density
from .density3d import multidensity3d

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One-shot routine to generate all plots used in the paper
# NOTE: will be slow, especially on the first run

def make_paper_plots():
    ''' A one-shot routine to make all the plots appearing in Cosyn/Freese/Sosa.
    This could take a long time to run, especially on the first call. Subsequent
    calls will run faster because deupack will create caches of the EMTFFs and
    their Bessel transforms, but the plots could still take time to render.
    '''
    group_comparison() # Figure 1
    D()                # Figure 2
    cbar()             # Figure 3
    antisymmetric()    # Figure 4
    mass_density()     # Figure 5
    s_d_interference() # Figure 6
    momentum_density() # Figure 7
    principal_axes()   # Figure 9
    pressure()         # Figure 10
    torsion()          # Figure 12
    forces()           # Figure 13
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EMTFF plots

def group_comparison():
    ''' Creates a six-panel figure with plots of the conserved symmetric EMTFFs.
    Compares the deuteron EMTFFs of the following works:
        - Cosyn, Freese and Sosa
          in preparation (this paper)
        - Freese and Cosyn
          Physical Review D 106 (2022) 114013
          Freese:2022yur
        - He and Zahed
          Physical Review C 110 (2024) 014312
          He:2024vzz
        - Panteleeva et al.
          Acta. Phys. Polon. B 56 (2025) 3-A19
          Panteleeva:2024abz
    '''
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = plt.subplot(nrows,ncols,1)
    ax_AT  = plt.subplot(nrows,ncols,2)
    ax_J   = plt.subplot(nrows,ncols,3)
    ax_DU  = plt.subplot(nrows,ncols,4)
    ax_DT1 = plt.subplot(nrows,ncols,5)
    ax_DT2 = plt.subplot(nrows,ncols,6)
    _group_comparison_panel(ax_AU,  'AU')
    _group_comparison_panel(ax_AT,  'AT')
    _group_comparison_panel(ax_J,   'J')
    _group_comparison_panel(ax_DU,  'DU')
    _group_comparison_panel(ax_DT1, 'DT1')
    _group_comparison_panel(ax_DT2, 'DT2')
    # Remove x axes from top three panels for economic use of space
    for ax in [ax_AU, ax_AT, ax_J]:
        ax.get_xaxis().set_visible(False)
    # Legend only needs to be put in one panel; use AU panel.
    legend = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    legend.get_frame().set_facecolor('#f8f8f8')
    # Transparent background so it looks nice in talk slides
    fig.patch.set_alpha(0)
    fig.savefig('group_comparisons.pdf')
    return

def D():
    ''' Creates three-panel figure for the D-like form factors.
    Each panel has four curves, comparing two choices of wave function:
        (1) AV18
        (2) CD Bonn
    and two sets of nucleon EMTFFs:
        (1) Broniowski & Ruiz Arriola (from Broniowski:2025ctl)
        (2) pointlike nucleons
    '''
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_DU  = plt.subplot(nrows,ncols,1)
    ax_DT1 = plt.subplot(nrows,ncols,2)
    ax_DT2 = plt.subplot(nrows,ncols,3)
    _4curve_panel(ax_DU,  'DU')
    _4curve_panel(ax_DT1, 'DT1')
    _4curve_panel(ax_DT2, 'DT2')
    # Legends in DU panel, since it's the first
    leg1 = ax_DU.legend(prop={'size': 24},loc=(0.02,0.7))
    leg1.get_frame().set_facecolor('#f8f8f8')
    ax_DU.add_artist(leg1)
    # Make custom legend handles
    legend_elements = [
            mpl.lines.Line2D([0], [0], linestyle='-',  color='black', lw=2.6, label=r'Dipole nucleons'),
            mpl.lines.Line2D([0], [0], linestyle='--', color='black', lw=2.6, label=r'Point nucleons')
            ]
    legend = ax_DU.legend(handles=legend_elements, prop = { 'size' : 27 }, loc=6)
    legend.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('D.pdf')
    return

def cbar():
    ''' Creates three-panel figure for the cbar-like form factors.
    Each panel has four curves, comparing two choices of wave function:
        (1) AV18
        (2) CD Bonn
    and two sets of nucleon EMTFFs:
        (1) Broniowski & Ruiz Arriola (from Broniowski:2025ctl)
        (2) pointlike nucleons
    '''
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_cU  = plt.subplot(nrows,ncols,1)
    ax_cT1 = plt.subplot(nrows,ncols,2)
    ax_cT2 = plt.subplot(nrows,ncols,3)
    _4curve_panel(ax_cU,  'cU')
    _4curve_panel(ax_cT1, 'cT1')
    _4curve_panel(ax_cT2, 'cT2')
    # Legends in cU panel, since it's the first
    leg1 = ax_cU.legend(prop={'size': 24},loc=2)
    leg1.get_frame().set_facecolor('#f8f8f8')
    ax_cU.add_artist(leg1)
    # Make custom legend handles
    legend_elements = [
            mpl.lines.Line2D([0], [0], linestyle='-',  color='black', lw=2.6, label=r'Dipole nucleons'),
            mpl.lines.Line2D([0], [0], linestyle='--', color='black', lw=2.6, label=r'Point nucleons')
            ]
    legend = ax_cU.legend(handles=legend_elements, prop = { 'size' : 27 }, loc=6)
    legend.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('cbar.pdf')
    return

def antisymmetric():
    ''' Creates two-panel figure for the antisymmetric form factors.
    Each panel has two curves, comparing a meson dominance form to
    pointlike nucleons.
    '''
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_S    = plt.subplot(nrows,ncols,1)
    ax_sbar = plt.subplot(nrows,ncols,2)
    _2curve_panel(ax_S,    'S')
    _2curve_panel(ax_sbar, 'sbar')
    # Legend in S panel, since it's the first
    legend = ax_S.legend(prop = { 'size' : 27 }, loc=1)
    legend.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('antisymmetric.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Three-dimensional density plots

def mass_density(nff='ba', wf='av18', nb=101, bmax=2):
    # Compute densities
    nb = 101; bmax = 2; nff = 'ba'; wf='av18'
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    M0 = D.mass_density(pol=0)
    M1 = D.mass_density(pol=1)
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'$m_j=0$', r'$m_j=\pm1$']
    # Use custom routine from density3d.py
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   M0, M1,
                   labels=labels,
                   clabel=r'Mass density (GeV/fm$^3$)',
                   decay=4, opacity=0.69, cmap=cmr.voltage_r,
                   projections=True, divergent=False, s=1
                   )
    fig.savefig('mass3D.pdf')
    return

def momentum_density(nff='ba', wf='av18', nb=101, bmax=2):
    # Compute densities
    nb = 101; bmax = 2; nff = 'ba'; wf='av18'
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    p = D.momentum_density(pol=1)
    f = D.flux_density(pol=1)
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'Momentum density ($m_j=1$)', r'Mass flux density ($m_j=1$)']
    # Use custom routine from density3d.py
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   p, f,
                   labels=labels,
                   clabel=r'$\phi$ projection of density (GeV/fm$^3$)',
                   decay=4, opacity=0.69, cmap=cmr.voltage_r,
                   projections=True, divergent=False, s=1
                   )
    fig.savefig('momentum3D.pdf')
    return

def pressure():
    # Fixed parameters for the visualization
    nff='ba'; wf='av18'; nb=101; bmax=2
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
    fig.savefig('pressure3D.pdf')
    return

def torsion():
    # Fixed parameters for the visualization
    nff='ba'; wf='av18'; nb=101; bmax=2
    # Get the torsion
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    s0 = D.torsion_shear(pol=0)
    s1 = D.torsion_shear(pol=1)
    # Make some labels
    labels = [ r'$m_j=0$', r'$m_j=\pm1$' ]
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   s0, s1,
                   labels=labels,
                   clabel=r'Torsion stress (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('torsion3D.pdf')
    return

def s_d_interference():
    # Fixed parameters for the visualization
    nff='ba'; nb=101; bmax=2
    wf_f = 'av18'
    wf_s = 'av18-s-only'
    wf_d = 'av18-d-only'
    # Get the full, S-only and D-only contributions to mass density
    D_F = Density(nff=nff, wf=wf_f, nb=nb, bmax=bmax)
    D_S = Density(nff=nff, wf=wf_s, nb=nb, bmax=bmax)
    D_D = Density(nff=nff, wf=wf_d, nb=nb, bmax=bmax)
    M0_F = D_F.mass_density(pol=0)
    M0_S = D_S.mass_density(pol=0)
    M0_D = D_D.mass_density(pol=0)
    M1_F = D_F.mass_density(pol=1)
    M1_S = D_S.mass_density(pol=1)
    M1_D = D_D.mass_density(pol=1)
    # Get interference from removing S-only and D-only
    M0_I = M0_F - M0_S - M0_D
    M1_I = M1_F - M1_S - M1_D
    # Make some labels
    labels = [
            r'S-wave ($m_j=0$)', r'D-wave ($m_j=0$)', r'Interference ($m_j=0$)',
            r'S-wave ($m_j=\pm1$)', r'D-wave ($m_j=\pm1$)', r'Interference ($m_j=\pm1$)'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D_F.x, D_F.x, D_F.x,
                   nrows, ncols,
                   M0_S, M0_D, M0_I, M1_S, M1_D, M1_I,
                   labels=labels,
                   clabel=r'Mass density contribution (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('s_d_interference.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Two-dimensional density/quiver/streamline plots

def principal_axes():
    # Parameters for this visualization (fixed)
    bmax = 1.6; nff='ba'; wf='av18'; nbq = 21; nbh = 101
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
    fig.savefig('principal_axes.pdf', bbox_inches="tight")
    return

def forces():
    # Fixed parameters
    nff = 'ba'; wf = 'av18'; bmax = 1.4; nb = 101
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
    fig.savefig('forces.pdf', bbox_inches="tight")
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities for EMTFF plots

_namelabel = {
        'AU'  : r'$A_U(\varDelta^2)$',
        'AT'  : r'$A_T(\varDelta^2)$',
        'J'   : r'$J(\varDelta^2)$',
        'DU'  : r'$D_U(\varDelta^2)$',
        'DT1' : r'$D_{T1}(\varDelta^2)$',
        'DT2' : r'$D_{T2}(\varDelta^2)$',
        'cU'  : r'$\bar{c}_U(\varDelta^2)$',
        'cT1' : r'$\bar{c}_{T1}(\varDelta^2)$',
        'cT2' : r'$\bar{c}_{T2}(\varDelta^2)$',
        'S'   : r'$S(\varDelta^2)$',
        'sbar': r'$\bar{s}(\varDelta^2)$'
        }

def _select_emtff(name, dl2, wf='av18', nff='ba'):
    if(name=='AU'):
        F = emtff.AU(  np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='AT'):
        F = emtff.AT(  np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='J'):
        F = emtff.J(   np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DU'):
        F = emtff.DU(  np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT1'):
        F = emtff.DT1( np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT2'):
        F = emtff.DT2( np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cU'):
        F = emtff.cU(  np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT1'):
        F = emtff.cT1( np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT2'):
        F = emtff.cT2( np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='S'):
        F = emtff.S(   np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='sbar'):
        F = emtff.sbar(np.sqrt(dl2), wf=wf, nff=nff)
    else:
        F = dl2 * 0
    return F

# Panel plots for EMTFFs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: there's a lot of repeated code; this could be modularized

def _4curve_panel(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    # 4 curve version
    F_ba_18 = _select_emtff(name, dl2, nff='hz',    wf='av18')
    F_pt_18 = _select_emtff(name, dl2, nff='point', wf='av18')
    F_ba_cd = _select_emtff(name, dl2, nff='hz',    wf='cdbonn')
    F_pt_cd = _select_emtff(name, dl2, nff='point', wf='cdbonn')
    ax.plot(dl2, F_ba_18, '-',  linewidth=2.6, color='tab:blue',label='AV18')#,   label=r'AV18 + BA')
    ax.plot(dl2, F_pt_18, '--', linewidth=2.6, color='tab:blue')#,   label=r'AV18 + point')
    ax.plot(dl2, F_ba_cd, '-',  linewidth=2.6, color='tab:orange',label='CDBonn')#, label=r'CDBonn + BA')
    ax.plot(dl2, F_pt_cd, '--', linewidth=2.6, color='tab:orange')#, label=r'CDBonn + point')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.5')
    if(name=='cU' or name=='DU' or name=='DT1'):
        textxy = (0.74,0.09)
    elif(name=='cT1'):
        textxy = (0.05,0.09)
    else:
        textxy = (0.05,0.09)
    ax.annotate(
            _namelabel[name], xy=textxy, xycoords='axes fraction',
            bbox=bbox
            )
    # Need to manually adjust the window for DT1 to avoid tick label overlap
    if(name=='DT1'):
        ax.set_ylim((-311,11))
    ax.set_xscale('log')
    ax.set_xlim((1e-6,10))
    return

def _3curve_panel(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    # 3 curve version
    F_domin = _select_emtff(name, dl2, nff='ba')
    F_holog = _select_emtff(name, dl2, nff='hz')
    F_point = _select_emtff(name, dl2, nff='point')
    ax.plot(dl2, F_domin, '-',  linewidth=2.6, color='tab:blue',   label=r'Meson dominance')
    ax.plot(dl2, F_holog, '--', linewidth=2.6, color='tab:green',  label=r'Holography')
    ax.plot(dl2, F_point, '-.', linewidth=2.6, color='tab:orange', label=r'Point nucleons')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.5')
    if(name=='cU' or name=='DU' or name=='DT1'):
        textxy = (0.74,0.09)
    elif(name=='cT1'):
        textxy = (0.05,0.09)
    else:
        textxy = (0.05,0.09)
    ax.annotate(
            _namelabel[name], xy=textxy, xycoords='axes fraction',
            bbox=bbox
            )
    # Need to manually adjust the window for DT1 to avoid tick label overlap
    if(name=='DT1'):
        ax.set_ylim((-311,11))
    ax.set_xscale('log')
    ax.set_xlim((1e-6,10))
    return

def _2curve_panel(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    # 3 curve version
    F_domin = _select_emtff(name, dl2, nff='ba')
    F_point = _select_emtff(name, dl2, nff='point')
    ax.plot(dl2, F_domin, '-',  linewidth=2.6, color='tab:blue',   label=r'Dipole nucleons')
    ax.plot(dl2, F_point, '--', linewidth=2.6, color='tab:orange', label=r'Point nucleons')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.5')
    if(name=='cU' or name=='DU' or name=='DT1'):
        textxy = (0.74,0.09)
    elif(name=='cT1'):
        textxy = (0.05,0.09)
    else:
        textxy = (0.05,0.09)
    ax.annotate(
            _namelabel[name], xy=textxy, xycoords='axes fraction',
            bbox=bbox
            )
    # Need to manually adjust the window for DT1 to avoid tick label overlap
    if(name=='DT1'):
        ax.set_ylim((-311,11))
    ax.set_xscale('log')
    ax.set_xlim((1e-6,10))
    return

def _group_comparison_panel(ax, name):
    # EMTFF from other papers
    df_wc = emtff.wim.make_wimffs()
    fc_hz = emtff.hz.make_hzffs()
    df_jp = emtff.pegg.make_peggffs()
    F_wc = df_wc[name]
    F_hz = fc_hz[name]
    F_jp = df_jp[name]
    dl2_wc = df_wc['Delta2']
    dl2_hz = fc_hz['Delta2']
    dl2_jp = df_jp['Delta2']
    # Our EMTFF
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F = _select_emtff(name, dl2, nff='hz') # Use HZ NFFs for apples-to-apples comparison
    # Use the Tableau Palette (default as of matplotlib v2),
    # since it was designed with accessibility in mind.
    ax.plot(dl2,    F,    '-',  linewidth=2.6, color='tab:blue',  label=r'Ours')
    ax.plot(dl2_wc, F_wc, '--', linewidth=2.6, color='tab:orange',label=r'Freese and Cosyn')
    ax.plot(dl2_hz, F_hz, '-.', linewidth=2.6, color='tab:green', label=r'He and Zahed')
    ax.plot(dl2_jp, F_jp, ':',  linewidth=2.6, color='tab:red',   label=r'Panteleeva \textsl{et al.}')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.5')
    if(name=='DU'):
        textxy = (0.74,0.08)
    else:
        textxy = (0.74,0.88)
    ax.annotate(
            _namelabel[name], xy=textxy, xycoords='axes fraction',
            bbox=bbox
            )
    ax.set_xscale('log')
    # Limit the window for DT1 and DT2 to improve visibility
    if(name=='DT1'):
        ax.set_ylim((-560,560))
    if(name=='DT2'):
        ax.set_ylim((-0.17,0.59))
    ax.set_xlim((1e-6,10))
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

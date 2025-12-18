import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import emtff
from ..density import *
from ..emtff.nucleon.chooser import choose_nff
from ..wf.av18 import VmeanU, VmeanT

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

from .density3d import density3d, multidensity3d

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Force plot (since it's under active development

def forces(
        bmax=1,
        nb=101,
        nbq=21,
        wf='av18',
        nff='point',
        decor='quivers',
        pols='01'
        ):
    if(decor=='quivers'):
        _forces_via_quivers(bmax=bmax, nff=nff, wf=wf, nbq=nbq, nbh=nb, pols=pols)
    elif(decor=='streamlines'):
        _forces_via_streamlines(bmax=bmax, nff=nff, wf=wf, nb=nb)
    else:
        raise ValueError("{} is not a valid decor option.".format(decor))
    return

# Two ideas for showing force directions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _forces_via_streamlines(bmax=1, nff='ba', wf='av18', nb=101, pols='01'):
    # TODO ... work in progress
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    # Prepare figure
    nrows,ncols=1,2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    ax0 = axes[0]
    ax1 = axes[1]
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1)
    if(pols=='01'):
        _ = _force_panel_stream(ax0, D, pol=0, norm=norm, label=r'$m_j=0$')
        _ = _force_panel_stream(ax1, D, pol=1, norm=norm, label=r'$m_j=\pm 1$')
    elif(pols=='UT'):
        _ = _force_panel_stream(ax0, D, pol='U', norm=norm, label=r'Unpolarized')
        _ = _force_panel_stream(ax1, D, pol='T', norm=norm, label=r'Tensor-polarized')
    else:
        raise ValueError("{} is not a valid pols option.".format(pols))
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

def _forces_via_quivers(bmax=1, nff='ba', wf='av18', nbq=21, nbh=101, pols='01'):
    # TODO ... work in progress
    # Density objects for quiver (small) and heat map (large)
    Dq = Density(nff=nff, wf=wf, bmax=bmax, nb=nbq)
    Dh = Density(nff=nff, wf=wf, bmax=bmax, nb=nbh)
    # Prepare figure
    nrows,ncols=1,2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8.4,nrows*7.11), layout='constrained')
    ax0 = axes[0]
    ax1 = axes[1]
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1)
    if(pols=='01'):
        _ = _force_panel(ax0, Dq, Dh, pol=0, norm=norm, label=r'$m_j=0$')
        _ = _force_panel(ax1, Dq, Dh, pol=1, norm=norm, label=r'$m_j=\pm 1$')
    elif(pols=='UT'):
        _ = _force_panel(ax0, Dq, Dh, pol='U', norm=norm, label=r'Unpolarized')
        _ = _force_panel(ax1, Dq, Dh, pol='T', norm=norm, label=r'Tensor-polarized')
    else:
        raise ValueError("{} is not a valid pols option.".format(pols))
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

# Helpers for both methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _singlequiver(ax, x, y, vx, vy):
    quiver_kwargs_white = {
            'color' : 'white',
            'angles' : 'xy',
            'scale_units' : 'xy',
            'pivot' : 'mid',
            'scale' : 3/4*x.shape[0]/x.max(),
            'width' : 1/(10*x.shape[0])
            }
    quiver_kwargs_black = {
            'color' : 'black',
            'angles' : 'xy',
            'scale_units' : 'xy',
            'pivot' : 'mid',
            'scale' : x.shape[0]/x.max(),
            'width' : 0.75/(10*x.shape[0])
            }
    ax.quiver(x, y,  vx.T,  vy.T, **quiver_kwargs_white)
    ## ax.quiver(x, y,  vx.T,  vy.T, **quiver_kwargs_black)
    return

def _force_panel(ax, Dq, Dh, pol, norm, label):
    # First, calculate the quivers for force directions
    bq = Dq.x
    nbq = bq.shape[0]
    # Get force vectors for the quiver plot ... sliced down to y=0
    fr = Dq.radial_force(pol=pol)[:,nbq//2,:]
    fθ = Dq.polar_force( pol=pol)[:,nbq//2,:]
    # Pull out the angular dependence
    theta = Dq.theta[:,nbq//2,:]
    phi = Dq.phi[:,nbq//2,:]
    # Force in Cartesian coordinates
    fz = (fr*np.cos(theta) - fθ*np.sin(theta))
    fx = (fr*np.sin(theta) + fθ*np.cos(theta)) * np.cos(phi)
    # Divide out magnitude to get unit vector in desired direction
    f = np.sqrt(fx**2 + fz**2)
    fz /= f
    fx /= f
    # Next, get the fine-grained force magnitude for the heat map
    bh = Dh.x
    nbh = bh.shape[0]
    fr = Dh.radial_force(pol=pol)[:,nbh//2,:]
    fθ = Dh.polar_force( pol=pol)[:,nbh//2,:]
    f = np.sqrt(fr**2 + fθ**2)
    # Plot the heat map
    c = ax.pcolormesh(bh, bh, f.T, norm=norm, cmap=cmr.voltage_r, shading='gouraud')
    _singlequiver(ax, bq, bq, fx, fz)
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
    ## # Divide out magnitude to get unit vector in desired direction
    ## fz /= f
    ## fx /= f
    # Next, get the fine-grained force magnitude for the heat map
    # Plot the heat map
    c = ax.pcolormesh(b, b, f.T, norm=norm, cmap=cmr.voltage_r, shading='gouraud')
    # Plot the streamlines ... TODO
    s = ax.streamplot(b, b, fx.T, fz.T,
                      color='white',
                      arrowsize=1.7, arrowstyle='->',
                      broken_streamlines=False,
                      density=0.5
                      )
    # Tune the alphas for better visibility
    s.lines.set_alpha(0.31)
    for x in ax.get_children():
        if type(x)==mpl.patches.FancyArrowPatch:
            x.set_alpha(0.53)
    # Finish up
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax.annotate(label, xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax.set_xlabel(r'$x$ (fm)')
    ax.set_ylabel(r'$z$ (fm)')
    return c


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Routines to make specific plots

def plot_nff_comparisons():
    nrows,ncols=2,6
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = plt.subplot(nrows,ncols,1)
    ax_AT  = plt.subplot(nrows,ncols,2)
    ax_J   = plt.subplot(nrows,ncols,3)
    ax_DU  = plt.subplot(nrows,ncols,4)
    ax_DT1 = plt.subplot(nrows,ncols,5)
    ax_DT2 = plt.subplot(nrows,ncols,6)
    ax_cU  = plt.subplot(nrows,ncols,7)
    ax_cT1 = plt.subplot(nrows,ncols,8)
    ax_cT2 = plt.subplot(nrows,ncols,9)
    ax_S   = plt.subplot(nrows,ncols,10)
    ax_sbar   = plt.subplot(nrows,ncols,11)
    plot_one_nff(ax_AU,  'AU')
    plot_one_nff(ax_AT,  'AT')
    plot_one_nff(ax_J,   'J')
    plot_one_nff(ax_DU,  'DU')
    plot_one_nff(ax_DT1, 'DT1')
    plot_one_nff(ax_DT2, 'DT2')
    plot_one_nff(ax_cU,  'cU')
    plot_one_nff(ax_cT1,  'cT1')
    plot_one_nff(ax_cT2,  'cT2')
    plot_one_nff(ax_S,  'S')
    plot_one_nff(ax_sbar,  'sbar')
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('nff_comparisons.pdf')
    return

def plot_wf_comparisons():
    nrows,ncols=2,6
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = plt.subplot(nrows,ncols,1)
    ax_AT  = plt.subplot(nrows,ncols,2)
    ax_J   = plt.subplot(nrows,ncols,3)
    ax_DU  = plt.subplot(nrows,ncols,4)
    ax_DT1 = plt.subplot(nrows,ncols,5)
    ax_DT2 = plt.subplot(nrows,ncols,6)
    ax_cU  = plt.subplot(nrows,ncols,7)
    ax_cT1 = plt.subplot(nrows,ncols,8)
    ax_cT2 = plt.subplot(nrows,ncols,9)
    ax_S   = plt.subplot(nrows,ncols,10)
    ax_sbar   = plt.subplot(nrows,ncols,11)
    plot_one_wf(ax_AU,  'AU')
    plot_one_wf(ax_AT,  'AT')
    plot_one_wf(ax_J,   'J')
    plot_one_wf(ax_DU,  'DU')
    plot_one_wf(ax_DT1, 'DT1')
    plot_one_wf(ax_DT2, 'DT2')
    plot_one_wf(ax_cU,  'cU')
    plot_one_wf(ax_cT1, 'cT1')
    plot_one_wf(ax_cT2, 'cT2')
    plot_one_wf(ax_S,   'S')
    plot_one_wf(ax_sbar,  'sbar')
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('wf_comparisons.pdf')
    return

def plot_group_comparisons():
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = plt.subplot(nrows,ncols,1)
    ax_AT  = plt.subplot(nrows,ncols,2)
    ax_J   = plt.subplot(nrows,ncols,3)
    ax_DU  = plt.subplot(nrows,ncols,4)
    ax_DT1 = plt.subplot(nrows,ncols,5)
    ax_DT2 = plt.subplot(nrows,ncols,6)
    plot_one_group(ax_AU,  'AU')
    plot_one_group(ax_AT,  'AT')
    plot_one_group(ax_J,   'J')
    plot_one_group(ax_DU,  'DU')
    plot_one_group(ax_DT1, 'DT1')
    plot_one_group(ax_DT2, 'DT2')
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('group_comparisons.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Common utilities for plots

namelabel = {
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
        'sbar'   : r'$\bar{s}(\varDelta^2)$'
        }

def select_emtff(name, dl2, wf='av18', nff='mit'):
    if(name=='AU'):
        F = emtff.AU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='AT'):
        F = emtff.AT(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='J'):
        F = emtff.J(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DU'):
        F = emtff.DU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT1'):
        F = emtff.DT1(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT2'):
        F = emtff.DT2(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cU'):
        F = emtff.cU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT1'):
        F = emtff.cT1(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT2'):
        F = emtff.cT2(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='S'):
        F = emtff.S(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='sbar'):
        F = emtff.sbar(np.sqrt(dl2), wf=wf, nff=nff)
    else:
        F = dl2 * 0
    return F

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Methods to plot curves on one panel

def plot_one_nff(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F_mit = select_emtff(name, dl2, nff='mit')
    F_hz  = select_emtff(name, dl2, nff='hz')
    F_ba  = select_emtff(name, dl2, nff='ba')
    F_point  = select_emtff(name, dl2, nff='point')
    # Plot
    ax.plot(dl2, F_mit, '-',  linewidth=2, color='xkcd:true green',  label=r'MIT')
    ax.plot(dl2, F_hz,  '--', linewidth=2, color='xkcd:rich purple', label=r'He and Zahed')
    ax.plot(dl2, F_ba,  '-.', linewidth=2, color='xkcd:cobalt',      label=r'Broniowski and Ruiz Arriola')
    ax.plot(dl2, F_point,  '--', linewidth=2, color='xkcd:red',      label=r'Point')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(namelabel[name])
    ax.set_xscale('log')
    ax.set_xlim((1e-6,10))
    return

def plot_one_wf(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F_av18  = select_emtff(name, dl2, wf='av18')
    F_paris = select_emtff(name, dl2, wf='paris')
    # Plot
    ax.plot(dl2, F_av18,  '-',  linewidth=2, color='xkcd:true green',  label=r'AV18')
    ax.plot(dl2, F_paris, '--', linewidth=2, color='xkcd:rich purple', label=r'Paris')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(namelabel[name])
    ax.set_xscale('log')
    ax.set_xlim((1e-6,10))
    return

def plot_one_group(ax, name):
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
    F = select_emtff(name, dl2, nff='hz') # Use HZ NFFs for apples-to-apples comparison
    # Plot
    ax.plot(dl2,    F,    '-',  linewidth=2, color='xkcd:true green', label=r'Ours')
    ax.plot(dl2_wc, F_wc, '--', linewidth=2, color='xkcd:rich purple',label=r'Freese and Cosyn')
    ax.plot(dl2_hz, F_hz, '-.', linewidth=2, color='xkcd:cobalt', label=r'He and Zahed')
    ax.plot(dl2_jp, F_jp, ':',  linewidth=2, color='xkcd:coral',  label=r'Panteleva \textsl{et al.}')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(namelabel[name])
    ax.set_xscale('log')
    if(name=='DT1'):
        ax.set_ylim((-560,560))
        ##ax.plot(dl2, dl2*0 + emtff.DT1_zero(nff='hz'), linewidth=1, color='black') # test forward limit
    if(name=='DT2'):
        ax.set_ylim((-0.69,1.37))
        ##ax.plot(dl2, dl2*0 + emtff.DT2_zero(), linewidth=1, color='black') # test forward limit
    ax.set_xlim((1e-6,10))
    return

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import mff
from ..density import *
from ..mff.nucleon.chooser import choose_nff
from ..wf.av18 import VmeanU, VmeanT

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

from .density3d import density3d, multidensity3d

# Testing stuff
import time

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

def select_mff(name, dl2, wf='av18', nff='mit'):
    if(name=='AU'):
        F = mff.AU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='AT'):
        F = mff.AT(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='J'):
        F = mff.J(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DU'):
        F = mff.DU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT1'):
        F = mff.DT1(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='DT2'):
        F = mff.DT2(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cU'):
        F = mff.cU(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT1'):
        F = mff.cT1(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='cT2'):
        F = mff.cT2(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='S'):
        F = mff.S(np.sqrt(dl2), wf=wf, nff=nff)
    elif(name=='sbar'):
        F = mff.sbar(np.sqrt(dl2), wf=wf, nff=nff)
    else:
        F = dl2 * 0
    return F

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Methods to plot curves on one panel

def plot_one_nff(ax, name):
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F_mit = select_mff(name, dl2, nff='mit')
    F_hz  = select_mff(name, dl2, nff='hz')
    F_ba  = select_mff(name, dl2, nff='ba')
    F_point  = select_mff(name, dl2, nff='point')
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
    F_av18  = select_mff(name, dl2, wf='av18')
    F_paris = select_mff(name, dl2, wf='paris')
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
    # MFF from other papers
    df_wc = mff.wim.make_wimffs()
    fc_hz = mff.hz.make_hzmffs()
    df_jp = mff.pegg.make_peggmffs()
    F_wc = df_wc[name]
    F_hz = fc_hz[name]
    F_jp = df_jp[name]
    dl2_wc = df_wc['Delta2']
    dl2_hz = fc_hz['Delta2']
    dl2_jp = df_jp['Delta2']
    # Our MFF
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F = select_mff(name, dl2, nff='hz') # Use HZ NFFs for apples-to-apples comparison
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
        ##ax.plot(dl2, dl2*0 + mff.DT1_zero(nff='hz'), linewidth=1, color='black') # test forward limit
    if(name=='DT2'):
        ax.set_ylim((-0.69,1.37))
        ##ax.plot(dl2, dl2*0 + mff.DT2_zero(), linewidth=1, color='black') # test forward limit
    ax.set_xlim((1e-6,10))
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot nucleon MFFs...

def plot_DN():
    k = np.linspace(0, 1, 101)
    dl2 = k**2
    _, _, F_mit, *_ = choose_nff('mit')
    _, _, F_hz,  *_ = choose_nff('hz')
    _, _, F_ba,  *_ = choose_nff('ba')
    D_mit = F_mit(k)
    D_hz  = F_hz(k)
    D_ba  = F_ba(k)
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, D_mit, '-',  linewidth=2, color='xkcd:true green',  label=r'MIT')
    ax.plot(dl2, D_hz,  '--', linewidth=2, color='xkcd:rich purple', label=r'He and Zahed')
    ax.plot(dl2, D_ba,  '-.', linewidth=2, color='xkcd:cobalt',      label=r'Broniowski and Ruiz Arriola')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$D_N(\varDelta^2)$')
    l = ax.legend(prop = { 'size' : 27 }, loc=2)
    fig.patch.set_alpha(0)
    fig.savefig('DN.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plots for internal comparisons

def plot_cT2():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    cT2_Adam  = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2')
    cT2_Alan  = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Alan')
    # cT2_Adam3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Adam3')
    # cT2_Alan3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Alan3')
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, cT2_Adam,  '-',  linewidth=2, color='xkcd:true green', label=r'Adam')
    ax.plot(dl2, cT2_Alan,  '--', linewidth=2, color='xkcd:rich purple',label=r'Alan')
    # ax.plot(dl2, cT2_Adam3, '-.', linewidth=2, color='xkcd:cobalt',     label=r'Adam3')
    # ax.plot(dl2, cT2_Alan3, ':',  linewidth=2, color='xkcd:coral',      label=r'Alan3')
    #
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$\bar{c}_{T2}(\varDelta^2)$')
    l = ax.legend(prop = { 'size' : 27 }, loc=4)
    fig.patch.set_alpha(0)
    fig.savefig('cT2.pdf')
    return

def plot_cT1():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    cT1_Adam = mff.cT1(np.sqrt(dl2), nff='point', formula='cT1')
    cT1_Alan = mff.cT1(np.sqrt(dl2), nff='point', formula='cT1Alan')
    cT1_alnt = mff.cT1(np.sqrt(dl2), nff='point', formula='alannote')
    cT1_papr = mff.cT1(np.sqrt(dl2), nff='point', formula='paper')
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, cT1_Adam, '-',  linewidth=2.6, label=r"Adam's code")
    ax.plot(dl2, cT1_Alan, '--', linewidth=2.6, label=r"Alan's code")
    ax.plot(dl2, cT1_alnt, '-.', linewidth=2.6, label=r"Alan's note")
    ax.plot(dl2, cT1_papr, ':',  linewidth=2.6, label=r"Paper formula")
    #
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$\bar{c}_{T1}(\varDelta^2)$')
    ax.set_xscale('log')
    l = ax.legend(prop = { 'size' : 27 }, loc=1)
    fig.patch.set_alpha(0)
    fig.savefig('cT1.pdf')
    return

def plot_sbar():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    sbar_Adam  = mff.sbar(np.sqrt(dl2), nff='point', formula='sbar')
    sbar_Alan  = mff.sbar(np.sqrt(dl2), nff='point', formula='sbarAlan')
    # cT2_Adam3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Adam3')
    # cT2_Alan3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Alan3')
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, sbar_Adam,  '-',  linewidth=2, color='xkcd:true green', label=r'Adam')
    ax.plot(dl2, sbar_Alan,  '--', linewidth=2, color='xkcd:rich purple',label=r'Alan')
    #
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$\bar{s}(\varDelta^2)$')
    l = ax.legend(prop = { 'size' : 27 }, loc=4)
    fig.patch.set_alpha(0)
    fig.savefig('sbar.pdf')
    return

def plot_J():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    J_form1 = mff.J(np.sqrt(dl2), nff='point', formula='form1')
    J_form2 = mff.J(np.sqrt(dl2), nff='point', formula='form2')
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, J_form1,  '-',  linewidth=2, color='xkcd:true green', label=r'Form 1')
    ax.plot(dl2, J_form2,  '--', linewidth=2, color='xkcd:rich purple',label=r'Form 2')
    #
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$J(\varDelta^2)$')
    ax.set_xscale('log')
    l = ax.legend(prop = { 'size' : 27 }, loc=1)
    fig.patch.set_alpha(0)
    fig.savefig('J.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2D density or quiver plots

def doublequiver(ax, x, y, vx, vy):
    quiver_kwargs_white = {
            'color' : 'white',
            'angles' : 'xy',
            'scale_units' : 'xy',
            'pivot' : 'tail',
            'scale' : x.shape[0]/x.max(),
            'width' : 0.004
            }
    quiver_kwargs_black = {
            'color' : 'black',
            'angles' : 'xy',
            'scale_units' : 'xy',
            'pivot' : 'tail',
            'scale' : 5/4*x.shape[0]/x.max(),
            'width' : 0.003
            }
    ax.quiver(x, y,  vx.T,  vy.T, **quiver_kwargs_white)
    ax.quiver(x, y, -vx.T, -vy.T, **quiver_kwargs_white)
    ax.quiver(x, y,  vx.T,  vy.T, **quiver_kwargs_black)
    ax.quiver(x, y, -vx.T, -vy.T, **quiver_kwargs_black)
    return

def eigenvectors(nff='ba', wf='av18'):
    # Parameters for this visualization (fixed)
    bmax = 2
    # Quiver plot calculations
    nbq = 21
    Dq = Density(nff=nff, wf=wf, bmax=bmax, nb=nbq)
    bq = Dq.x
    Xp0, Yp0, Zp0 = Dq.e_plus( pol=0)
    Xp1, Yp1, Zp1 = Dq.e_plus( pol=1)
    Xm0, Ym0, Zm0 = Dq.e_minus(pol=0)
    Xm1, Ym1, Zm1 = Dq.e_minus(pol=1)
    # Slice at y=0
    xp0 = Xp0[:,nbq//2,:]
    zp0 = Zp0[:,nbq//2,:]
    xp1 = Xp1[:,nbq//2,:]
    zp1 = Zp1[:,nbq//2,:]
    xm0 = Xm0[:,nbq//2,:]
    zm0 = Zm0[:,nbq//2,:]
    xm1 = Xm1[:,nbq//2,:]
    zm1 = Zm1[:,nbq//2,:]
    # Density heat map calculations
    nbh = 101
    Dh = Density(nff=nff, wf=wf, bmax=bmax, nb=nbh)
    bh = Dh.x
    PLS0 = Dh.pseudoradial_pressure( pol=0)
    MIN0 = Dh.pseudolateral_pressure(pol=0)
    PLS1 = Dh.pseudoradial_pressure( pol=1)
    MIN1 = Dh.pseudolateral_pressure(pol=1)
    # Slice at y=0
    pls0 = PLS0[:,nbh//2,:]
    min0 = MIN0[:,nbh//2,:]
    pls1 = PLS1[:,nbh//2,:]
    min1 = MIN1[:,nbh//2,:]
    # Prepare figure
    nrows,ncols=2,2
    fig = plt.figure(figsize=(ncols*6,nrows*6), layout='constrained')
    ax0p = plt.subplot(nrows,ncols,1,aspect='equal')
    ax0m = plt.subplot(nrows,ncols,2,aspect='equal')
    ax1p = plt.subplot(nrows,ncols,3,aspect='equal')
    ax1m = plt.subplot(nrows,ncols,4,aspect='equal')
    # Heat maps ... NOTE: need transpose of mapped values!
    vmax = np.max([abs(pls0).max(), abs(min0).max(), abs(pls1).max(), abs(min1).max()])
    c0p = ax0p.pcolormesh(bh, bh, pls0.T, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    c0m = ax0m.pcolormesh(bh, bh, min0.T, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    c1p = ax1p.pcolormesh(bh, bh, pls1.T, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    c1m = ax1m.pcolormesh(bh, bh, min1.T, vmin=-vmax, vmax=vmax, cmap=cmr.fusion_r, shading='gouraud')
    # Quivers ... NOTE: need to take transpose of values for correct orientation!!!!
    doublequiver(ax0p, bq, bq, xp0, zp0)
    doublequiver(ax0m, bq, bq, xm0, zm0)
    doublequiver(ax1p, bq, bq, xp1, zp1)
    doublequiver(ax1m, bq, bq, xm1, zm1)
    # Finish up
    for ax in [ax0p, ax0m, ax1p, ax1m]:
        ax.set_xlabel(r'$x$ (fm)')
        ax.set_ylabel(r'$z$ (fm)')
    bbox = dict(facecolor='#f8f8f8', alpha=0.86, edgecolor='gray', boxstyle='round,pad=0.5')
    textxy = (0.05,0.09)
    ax0p.annotate(r'Pseudoradial,  $m_j=0$', xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax0m.annotate(r'Pseudolateral, $m_j=0$', xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax1p.annotate(r'Pseudoradial,  $m_j=1$', xy=textxy, xycoords='axes fraction', bbox=bbox)
    ax1m.annotate(r'Pseudolateral, $m_j=1$', xy=textxy, xycoords='axes fraction', bbox=bbox)
    fig.patch.set_alpha(0)
    fig.savefig('eigenvectors_{}_{}.pdf'.format(wf,nff))
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3D density plots

def plot_mass_3d(nff='ba', wf='av18', nb=101, bmax=2):
    t_0 = time.time()
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    M0 = D.mass_density(pol=0)
    M1 = D.mass_density(pol=1)
    t_A = time.time()
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'$m_j=0$', r'$m_j=1$']
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   M0, M1,
                   labels=labels,
                   clabel=r'Mass densities (GeV/fm$^3$)',
                   decay=4, opacity=0.69, cmap=cmr.voltage_r,
                   projections=True, divergent=False, s=1)
    t_B = time.time()
    fig.savefig('mass3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    t_C = time.time()
    print("Time to calculate mass densities: {:.3f} s".format(t_A-t_0))
    print("Time to plot mass densities:      {:.3f} s".format(t_B-t_A))
    print("Time to save plots:               {:.3f} s".format(t_C-t_B))
    return

def plot_momentum_3d(nff='ba', wf='av18', nb=101, bmax=2.0):
    D = Density(nff=nff, wf=wf, bmax=bmax, nb=nb)
    p = D.momentum_density(pol=1)
    f = D.flux_density(pol=1)
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'Momentum density ($m_j=1$)', r'Mass flux density ($m_j=1$)']
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   p, f,
                   labels=labels,
                   clabel=r'Density (GeV/fm$^3$)',
                   decay=4, opacity=0.69, cmap=cmr.voltage_r,
                   projections=True, divergent=False, s=1)
    fig.savefig('momentum3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    return

def plot_normal_stress_3d(nff='ba', wf='av18', nb=101, bmax=2):
    t_0 = time.time()
    # Load or create the stresses
    ###b = np.linspace(-bmax, bmax, nb)
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    pr0 = D.radial_pressure(   pol=0)
    pθ0 = D.lateral_pressure(  pol=0)
    pφ0 = D.azimuthal_pressure(pol=0)
    pr1 = D.radial_pressure(   pol=1)
    pθ1 = D.lateral_pressure(  pol=1)
    pφ1 = D.azimuthal_pressure(pol=1)
    t_A = time.time()
    # Make some labels
    labels = [
            r'$m_j=0$, radial pressure',
            r'$m_j=0$, lateral pressure',
            r'$m_j=0$, azimuthal pressure',
            r'$m_j=1$, radial pressure',
            r'$m_j=1$, lateral pressure',
            r'$m_j=1$, azimuthal pressure'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr0, pθ0, pφ0, pr1, pθ1, pφ1,
                   labels=labels,
                   clabel=r'Normal stresses (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    t_B = time.time()
    fig.savefig('normal_stress3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    t_C = time.time()
    print("Time to calculate (or load) pressures: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:                {:.3f} s".format(t_B-t_A))
    print("Time to save plots:                    {:.3f} s".format(t_C-t_B))
    return

def plot_shear_stress_3d(nff='ba', wf='av18', nb=101, bmax=2):
    t_0 = time.time()
    # Load or create the stresses
    ###b = np.linspace(-bmax, bmax, nb)
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    s= D.symmetric_shear(pol='T')
    t_A = time.time()
    # Make some labels
    labels = [
            r'symmetric shear $(r,\theta)$'
            ]
    # Prepare figure
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   s,
                   labels=labels,
                   clabel=r'Shear stress (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    t_B = time.time()
    fig.savefig('shear_stress3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    t_C = time.time()
    print("Time to calculate (or load) pressures: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:                {:.3f} s".format(t_B-t_A))
    print("Time to save plots:                    {:.3f} s".format(t_C-t_B))
    return

def plot_torsion_3d(nff='ba', wf='av18', nb=101, bmax=2):
    t_0 = time.time()
    # Load or create the stresses
    ###b = np.linspace(-bmax, bmax, nb)
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    s = D.torsion_shear(pol='T')
    s0 =  2/3*s
    s1 = -1/3*s
    t_A = time.time()
    # Make some labels
    labels = [
            r'$m_j=0$',
            r'$m_j=1$'
            ]
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   s0, s1,
                   labels=labels,
                   clabel=r'Torsional shear stress (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    t_B = time.time()
    fig.savefig('torsion3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    t_C = time.time()
    print("Time to calculate (or load) pressures: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:                {:.3f} s".format(t_B-t_A))
    print("Time to save plots:                    {:.3f} s".format(t_C-t_B))
    return

def plot_eigenpressures_3d(nff='ba', wf='av18', nb=101, bmax=2):
    t_0 = time.time()
    # Load or create the stresses
    ###b = np.linspace(-bmax, bmax, nb)
    D = Density(nff=nff, wf=wf, nb=nb, bmax=bmax)
    pr0 = D.pseudoradial_pressure( pol=0)
    pθ0 = D.pseudolateral_pressure(pol=0)
    pφ0 = D.azimuthal_pressure(    pol=0)
    pr1 = D.pseudoradial_pressure( pol=1)
    pθ1 = D.pseudolateral_pressure(pol=1)
    pφ1 = D.azimuthal_pressure(    pol=1)
    t_A = time.time()
    # Make some labels
    labels = [
            r'$m_j=0$, pseudoradial pressure',
            r'$m_j=0$, pseudolateral pressure',
            r'$m_j=0$, azimuthal pressure',
            r'$m_j=1$, pseudoradial pressure',
            r'$m_j=1$, pseudolateral pressure',
            r'$m_j=1$, azimuthal pressure'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr0, pθ0, pφ0, pr1, pθ1, pφ1,
                   labels=labels,
                   clabel=r'Eigenpressures (GeV/fm$^3$)',
                   decay=2, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    t_B = time.time()
    fig.savefig('eigenpressures3D_{}_{}_{:d}_{:.2f}.pdf'.format(wf,nff,nb,bmax))
    t_C = time.time()
    print("Time to calculate (or load) pressures: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:                {:.3f} s".format(t_B-t_A))
    print("Time to save plots:                    {:.3f} s".format(t_C-t_B))
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3D potential energy plot

def plot_potential_energy(nb=60, bmax=2.0):
    # Probability density
    D = Density(nff='point', wf='av18')
    b = np.linspace(-bmax, bmax, nb)
    MU = D.mass_3D_U(b,b,b)
    MT = D.mass_3D_T(b,b,b)
    P0 = (MU + 2/3*MT) / (2*mN) / 8 # Scale down by mass, factor 8 from r=2*b
    P1 = (MU - 1/3*MT) / (2*mN) / 8 # Scale down by mass, factor 8 from r=2*b
    # Potential ... remember here that r=2*b
    Y2 = make_Y2(b,b,b)
    rhoT = make_rhoT(b,b,b)
    harmonics = np.einsum('xyzij,xyzij->xyz', Y2, rhoT)
    x_, y_, z_ = np.meshgrid(2*b,2*b,2*b,indexing='ij')
    r_ = np.sqrt(x_**2 + y_**2 + z_**2 + 1e-12)
    VU = VmeanU(r_)
    VT = VmeanT(r_) * harmonics
    V0 = 1000*(VU + 2/3*VT) # convert from GeV to MeV
    V1 = 1000*(VU - 1/3*VT) # convert from GeV to MeV
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'$m_j=0$', r'$m_j=1$']
    multidensity3d(fig, b, b, b,
                   nrows, ncols,
                   V0/P0, V1/P1,
                   labels=labels,
                   clabel=r'$\psi^\dagger V_{\mathrm{eff}} \psi / (\psi^\dagger \psi)$ (MeV)',
                   decay=2, opacity=0.07, cmap=cmr.iceburn,
                   projections=False, divergent=True, s=1,
                   vmax=5)
    fig.savefig('potential.pdf')
    return

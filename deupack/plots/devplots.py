import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as py

from .. import mff
from ..mff.nucleon.chooser import choose_nff

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
py.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Routines to make specific plots

def plot_nff_comparisons():
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = py.subplot(nrows,ncols,1)
    ax_AT  = py.subplot(nrows,ncols,2)
    ax_J   = py.subplot(nrows,ncols,3)
    ax_DU  = py.subplot(nrows,ncols,4)
    ax_DT1 = py.subplot(nrows,ncols,5)
    ax_DT2 = py.subplot(nrows,ncols,6)
    #ax_cU  = py.subplot(nrows,ncols,7)
    #ax_cT1 = py.subplot(nrows,ncols,8)
    #ax_cT2 = py.subplot(nrows,ncols,9)
    #ax_S   = py.subplot(nrows,ncols,10)
    plot_one_nff(ax_AU,  'AU')
    plot_one_nff(ax_AT,  'AT')
    plot_one_nff(ax_J,   'J')
    plot_one_nff(ax_DU,  'DU')
    plot_one_nff(ax_DT1, 'DT1')
    plot_one_nff(ax_DT2, 'DT2')
    #plot_one_nff(ax_cU,  'cU')
    #plot_one_nff(ax_cT1, 'cT1')
    #plot_one_nff(ax_cT2, 'cT2')
    #plot_one_nff(ax_S,   'S')
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('nff_comparisons.pdf')
    return

def plot_wf_comparisons():
    nrows,ncols=2,5
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = py.subplot(nrows,ncols,1)
    ax_AT  = py.subplot(nrows,ncols,2)
    ax_J   = py.subplot(nrows,ncols,3)
    ax_DU  = py.subplot(nrows,ncols,4)
    ax_DT1 = py.subplot(nrows,ncols,5)
    ax_DT2 = py.subplot(nrows,ncols,6)
    ax_cU  = py.subplot(nrows,ncols,7)
    ax_cT1 = py.subplot(nrows,ncols,8)
    ax_cT2 = py.subplot(nrows,ncols,9)
    ax_S   = py.subplot(nrows,ncols,10)
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
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('wf_comparisons.pdf')
    return

def plot_grouo_comparisons():
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = py.subplot(nrows,ncols,1)
    ax_AT  = py.subplot(nrows,ncols,2)
    ax_J   = py.subplot(nrows,ncols,3)
    ax_DU  = py.subplot(nrows,ncols,4)
    ax_DT1 = py.subplot(nrows,ncols,5)
    ax_DT2 = py.subplot(nrows,ncols,6)
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
        'S'   : r'$S(\varDelta^2)$'
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
    # Plot
    ax.plot(dl2, F_mit, '-',  linewidth=2, color='xkcd:true green',  label=r'MIT')
    ax.plot(dl2, F_hz,  '--', linewidth=2, color='xkcd:rich purple', label=r'He and Zahed')
    ax.plot(dl2, F_ba,  '-.', linewidth=2, color='xkcd:cobalt',      label=r'Broniowski and Ruiz Arriola')
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
        ax.set_ylim((-560,56))
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
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
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

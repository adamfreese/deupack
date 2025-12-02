# paperplots.py
#
# Created 2025.12.02
#
# Routines for the plots included in our first deuteron stress paper.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import mff

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The group comparison plot

def group_comparison():
    ''' Creates a six-panel figure with plots of the conserved symmetric MFFs.
    Compares the deuteron MFFs of the following works:
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

# Utilities for the group comparison plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def _select_mff(name, dl2, wf='av18', nff='ba'):
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

def _group_comparison_panel(ax, name):
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
    F = _select_mff(name, dl2, nff='hz') # Use HZ NFFs for apples-to-apples comparison
    # Use the Tableau Palette (default as of v2),
    # since it was designed with accessibility in mind.
    ax.plot(dl2,    F,    '-',  linewidth=2.6, color='tab:blue',  label=r'Ours')
    ax.plot(dl2_wc, F_wc, '--', linewidth=2.6, color='tab:orange',label=r'Freese and Cosyn')
    ax.plot(dl2_hz, F_hz, '-.', linewidth=2.6, color='tab:green', label=r'He and Zahed')
    ax.plot(dl2_jp, F_jp, ':',  linewidth=2.6, color='tab:red',   label=r'Panteleeva \textsl{et al.}')
    # Line at zero to help guide the eye
    ax.plot(dl2, dl2*0, linewidth=1, color='tab:gray')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ##ax.set_ylabel(_namelabel[name])
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.5')
    if(name=='AU' or name=='AT' or name=='J'):
        textxy = (0.74,0.88)
    else:
        textxy = (0.74,0.08)
    ax.annotate(
            _namelabel[name], xy=textxy, xycoords='axes fraction',
            bbox=bbox
            )
    ax.set_xscale('log')
    # Limit the window for DT1 and DT2, in light of divergences in some cases
    if(name=='DT1'):
        ax.set_ylim((-560,560))
    if(name=='DT2'):
        ax.set_ylim((-0.69,1.37))
    ax.set_xlim((1e-6,10))
    return


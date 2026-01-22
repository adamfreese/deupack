import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import emtff

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross check of paper vs fast formulas

def cbar_check():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    dl = np.sqrt(dl2)
    cU_fast  = emtff.cU( dl, nff='point', formula='fast')
    cU_papr  = emtff.cU( dl, nff='point', formula='paper')
    cT1_fast = emtff.cT1(dl, nff='point', formula='fast')
    cT1_papr = emtff.cT1(dl, nff='point', formula='paper')
    cT2_fast = emtff.cT2(dl, nff='point', formula='fast')
    cT2_papr = emtff.cT2(dl, nff='point', formula='paper')
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    axU  = plt.subplot(nrows,ncols,1)
    axT1 = plt.subplot(nrows,ncols,2)
    axT2 = plt.subplot(nrows,ncols,3)
    axU.plot( dl2, cU_fast,  '-',  linewidth=2.6, label=r'Fast')
    axU.plot( dl2, cU_papr,  '--', linewidth=2.6, label=r'Paper')
    axT1.plot(dl2, cT1_fast, '-',  linewidth=2.6, label=r'Fast')
    axT1.plot(dl2, cT1_papr, '--', linewidth=2.6, label=r'Paper')
    axT2.plot(dl2, cT2_fast, '-',  linewidth=2.6, label=r'Fast')
    axT2.plot(dl2, cT2_papr, '--', linewidth=2.6, label=r'Paper')
    for ax in [axU, axT1, axT2]:
        ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
        ax.set_xscale('log')
    axU.set_ylabel(r'$\bar{c}_{U}(\varDelta^2)$')
    axT1.set_ylabel(r'$\bar{c}_{T1}(\varDelta^2)$')
    axT2.set_ylabel(r'$\bar{c}_{T2}(\varDelta^2)$')
    l = axU.legend(prop = { 'size' : 27 }, loc=2)
    fig.patch.set_alpha(0)
    fig.savefig('cbar_check.pdf')
    return


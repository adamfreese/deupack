import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import emtff
from ..wf.airy import dwf_airy
from ..wf.hydrogen import dwf_hydrogen

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check that Airy wave function behaves reasonably

def airy_check():
    dl2 = np.linspace(1e-4, 10, 666)
    dl = np.sqrt(dl2)
    eta_c = dwf_airy()
    # One-body form factors
    Aq = emtff.AU(dl, wf=eta_c, nff='point', impulse=True, string=False)
    Dq = emtff.DU(dl, wf=eta_c, nff='point', impulse=True, string=False)
    cq = emtff.cU(dl, wf=eta_c, nff='point', impulse=True, string=False)
    # String form factors
    AS = emtff.AU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    DS = emtff.DU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    cS = emtff.cU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    A = Aq + AS
    D = Dq + DS
    c = cq + cS
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    ax1.plot(dl2, Aq, '--', linewidth=2.6, label=r'Quark')
    ax1.plot(dl2, AS, ':',  linewidth=2.6, label=r'String')
    ax1.plot(dl2, A,  '-',  linewidth=2.6, label=r'Total')
    #
    ax2.plot(dl2, Dq, '--', linewidth=2.6, label=r'Quark')
    ax2.plot(dl2, DS, ':',  linewidth=2.6, label=r'String')
    ax2.plot(dl2, D,  '-',  linewidth=2.6, label=r'Total')
    #
    ax3.plot(dl2, cq, '--', linewidth=2.6, label=r'Quark')
    ax3.plot(dl2, cS, ':',  linewidth=2.6, label=r'String')
    ax3.plot(dl2, c,  '-',  linewidth=2.6, label=r'Total')
    #
    ax1.set_ylabel(r'$A(\varDelta^2)$')
    ax2.set_ylabel(r'$D(\varDelta^2)$')
    ax3.set_ylabel(r'$\bar{c}(\varDelta^2)$')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
        #ax.set_xscale('log')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    fig.patch.set_alpha(0)
    fig.savefig('airy_emtff.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check that hydrogen wave function behaves reasonably

def hydrogen_check():
    dl2 = np.geomspace(1e-12, 0.1, 666)
    dl = np.sqrt(dl2)
    H = dwf_hydrogen()
    # One-body form factors
    Aq = emtff.AU(dl, wf=H, nff='point', impulse=True, coulomb=False)
    Dq = emtff.DU(dl, wf=H, nff='point', impulse=True, coulomb=False)
    cq = emtff.cU(dl, wf=H, nff='point', impulse=True, coulomb=False)
    # Coulomb form factors
    AS = emtff.AU(dl, wf=H, nff='point', impulse=False, coulomb=True)
    DS = emtff.DU(dl, wf=H, nff='point', impulse=False, coulomb=True)
    cS = emtff.cU(dl, wf=H, nff='point', impulse=False, coulomb=True)
    A = Aq + AS
    D = Dq + DS
    c = cq + cS
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    cf = 1e6
    #
    ax1.plot(cf*dl2, Aq, '--', linewidth=2.6, label=r'Particle')
    ax1.plot(cf*dl2, AS, ':',  linewidth=2.6, label=r'Coulomb')
    ax1.plot(cf*dl2, A,  '-',  linewidth=2.6, label=r'Total')
    #
    ax2.plot(cf*dl2, Dq, '--', linewidth=2.6, label=r'Particle')
    ax2.plot(cf*dl2, DS, ':',  linewidth=2.6, label=r'Coulomb')
    ax2.plot(cf*dl2, D,  '-',  linewidth=2.6, label=r'Total')
    #
    ax3.plot(cf*dl2, cq, '--', linewidth=2.6, label=r'Particle')
    ax3.plot(cf*dl2, cS, ':',  linewidth=2.6, label=r'Coulomb')
    ax3.plot(cf*dl2, c,  '-',  linewidth=2.6, label=r'Total')
    #
    ax1.set_ylabel(r'$A(\varDelta^2)$')
    ax2.set_ylabel(r'$D(\varDelta^2)$')
    ax3.set_ylabel(r'$\bar{c}(\varDelta^2)$')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(r'$\varDelta^2$ (MeV$^2$)')
        ax.set_xscale('log')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    fig.patch.set_alpha(0)
    fig.savefig('hydrogen_emtff.pdf')
    return


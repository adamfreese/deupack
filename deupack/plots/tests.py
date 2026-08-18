import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.special import exp1 # for E1 test

from .. import emtff
from ..constants import hbar#, alphaQED
from ..wf.airy import dwf_airy
from ..wf.hydrogen import dwf_hydrogen
from ..wf.variational import vwf_cornell, vwf_yukawa

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test asymptotic expansion for E1

def E1_direct(z):
    z1 = z*(1+1j)
    z2 = z*(-1+1j)
    cterm1 = exp1(z1)*np.exp(z)
    cterm2 = exp1(z2)*np.exp(-z)
    result = np.imag(cterm1-cterm2)
    return result

def E1_asymptotic(z, n):
    result = -np.sin(z)/z
    if(n>=1):
        result += np.cos(z)/z**2
    if(n>=2):
        result += np.sin(z)/(2*z**3)
    return result

def E1_test(zmin=90, zmax=100):
    z = np.linspace(zmin, zmax, 8000)
    direct = E1_direct(z)
    asymp0 = E1_asymptotic(z, 0)
    asymp1 = E1_asymptotic(z, 1)
    asymp2 = E1_asymptotic(z, 2)
    #
    nrows, ncols = 1, 1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    ax.plot(z, direct, '-',  label=r'Direct implementation')
    ax.plot(z, asymp0, '--', label=r'Asymptotic ($n=0$)')
    ax.plot(z, asymp1, '-.', label=r'Asymptotic ($n=1$)')
    ax.plot(z, asymp2, ':',  label=r'Asymptotic ($n=2$)')
    #
    ax.set_xlabel(r'$z$')
    l = ax.legend(prop = { 'size' : 20 })
    fig.patch.set_alpha(0)
    fig.savefig('E1_test.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test variational ground state solver

def variational_test():
    r = np.linspace(0, 4, 666)
    # Wave functions
    wf2 = vwf_cornell(N=2)
    wf4 = vwf_cornell(N=4)
    wf6 = vwf_cornell(N=6)
    u2 = wf2.u(r)
    u4 = wf4.u(r)
    u6 = wf6.u(r)
    # Plots
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    # Plot wave functions
    ax1.plot(r, u2, ':',  color='tab:blue',   linewidth=2.6, label=r'$N=2$')
    ax1.plot(r, u4, '-.', color='tab:orange', linewidth=2.6, label=r'$N=4$')
    ax1.plot(r, u6, '--', color='tab:green',  linewidth=2.6, label=r'$N=6$')
    # Plot energy estimates
    ax2.plot(2, wf2.E, 'o', color='tab:blue')
    ax2.plot(4, wf4.E, 'o', color='tab:orange')
    ax2.plot(6, wf6.E, 'o', color='tab:green')
    # Labels etc
    ax1.set_ylabel(r'$u(r)$ (fm$^{-1/2}$)')
    ax1.set_xlabel(r'$r$ (fm)')
    ax2.set_ylabel(r'$E$ (GeV)')
    ax2.set_xlabel(r'$N$')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    l.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('variational_test.pdf')
    fig.savefig('variational_test.png')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check that Cornell wave function behaves reasonably

def cornell_check(N=4):
    dl2 = np.linspace(1e-4, 10, 666)
    dl = np.sqrt(dl2)
    eta_c = vwf_cornell(N=N)
    ##eta_c = var_wf_airy(N=N)
    # One-body form factors
    Aq = emtff.AU(dl, wf=eta_c, nff='point', impulse=True)
    Dq = emtff.DU(dl, wf=eta_c, nff='point', impulse=True)
    cq = emtff.cU(dl, wf=eta_c, nff='point', impulse=True)
    # String form factors
    AS = emtff.AU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    DS = emtff.DU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    cS = emtff.cU(dl, wf=eta_c, nff='point', impulse=False, string=True)
    # Coulomb form factors
    AC = emtff.AU(dl, wf=eta_c, nff='point', impulse=False, coulomb=True)
    DC = emtff.DU(dl, wf=eta_c, nff='point', impulse=False, coulomb=True)
    cC = emtff.cU(dl, wf=eta_c, nff='point', impulse=False, coulomb=True)
    A = Aq + AS + AC
    D = Dq + DS + DC
    c = cq + cS + cC
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    ax1.plot(dl2, Aq, '--', linewidth=2.6, label=r'Quark')
    ax1.plot(dl2, AS, ':',  linewidth=2.6, label=r'String')
    ax1.plot(dl2, AC, '-.', linewidth=2.6, label=r'Coulomb')
    ax1.plot(dl2, A,  '-',  linewidth=2.6, label=r'Total')
    #
    ax2.plot(dl2, Dq, '--', linewidth=2.6, label=r'Quark')
    ax2.plot(dl2, DS, ':',  linewidth=2.6, label=r'String')
    ax2.plot(dl2, DC, '-.', linewidth=2.6, label=r'Coulomb')
    ax2.plot(dl2, D,  '-',  linewidth=2.6, label=r'Total')
    #
    ax3.plot(dl2, cq, '--', linewidth=2.6, label=r'Quark')
    ax3.plot(dl2, cS, ':',  linewidth=2.6, label=r'String')
    ax3.plot(dl2, cC, '-.', linewidth=2.6, label=r'Coulomb')
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
    fig.savefig('cornell_emtff.pdf')
    fig.patch.set_alpha(1)
    fig.savefig('cornell_emtff.png')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def D_integrand_cornell():
    r = np.linspace(0, 4, 666)
    wf = var_wf_cornell()
    u  = wf.u(r)
    u2 = wf.u2(r)
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(r, r**2*u*u2)
    #
    ax.plot(r, r*0, '-', linewidth=1, color='tab:gray')
    # Classical barrier
    E = wf.Efm
    sigma = wf.sigma
    alpha = wf.alpha
    r0 = E/(2*sigma) + np.sqrt(E**2/(4*sigma**2) + alpha/sigma)
    ymin, ymax = ax.get_ylim()
    ax.vlines(r0, ymin, ymax)
    ax.set_ylim((ymin,ymax))
    #
    fig.savefig('D_integrand.pdf')
    return

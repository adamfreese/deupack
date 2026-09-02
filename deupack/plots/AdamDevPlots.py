import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.special import exp1 # for E1 test
from scipy.integrate import quad

from .. import emtff
from ..constants import hbar, alphaQED
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
# Test variational ground state solver

def variational_test():
    r = np.linspace(0, 4, 666)
    # Wave functions
    wf2 = vwf_yukawa(N=2)
    wf4 = vwf_yukawa(N=4)
    wf6 = vwf_yukawa(N=6)
    wf8 = vwf_yukawa(N=8)
    u2 = wf2.u(r)
    u4 = wf4.u(r)
    u6 = wf6.u(r)
    u8 = wf8.u(r)
    # Plots
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    # Plot wave functions
    ax1.plot(r, u2, ':',  color='tab:blue',   linewidth=2.6, label=r'$N=2$')
    ax1.plot(r, u4, '-.', color='tab:orange', linewidth=2.6, label=r'$N=4$')
    ax1.plot(r, u6, '--', color='tab:green',  linewidth=2.6, label=r'$N=6$')
    ax1.plot(r, u8, '-',  color='tab:purple', linewidth=2.6, label=r'$N=8$')
    # Plot energy estimates
    ax2.plot(2, wf2.E, 'o', color='tab:blue')
    ax2.plot(4, wf4.E, 'o', color='tab:orange')
    ax2.plot(6, wf6.E, 'o', color='tab:green')
    ax2.plot(8, wf6.E, 'o', color='tab:purple')
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
# Check that wave function with Yukawa potential behaves reasonably

def yukawa_check():
    dl2 = np.geomspace(1e-5, 100, 666)
    dl = np.sqrt(dl2)
    H = vwf_yukawa()
    # One-body form factors
    Aq = emtff.AU(dl, wf=H, nff='point', impulse=True)
    Dq = emtff.DU(dl, wf=H, nff='point', impulse=True)
    cq = emtff.cU(dl, wf=H, nff='point', impulse=True)
    # Field form factors
    field = {
            'g1' : np.sqrt(4*np.pi*H.alpha),
            'g2' : np.sqrt(4*np.pi*H.alpha),
            'mf' : H.mu,
            's'  : 0
            }
    Ag = emtff.AU(dl, wf=H, nff='point', impulse=False, field=field)
    Dg = emtff.DU(dl, wf=H, nff='point', impulse=False, field=field)
    cg = emtff.cU(dl, wf=H, nff='point', impulse=False, field=field)
    A = Aq + Ag
    D = Dq + Dg
    c = cq + cg
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    ax1.plot(dl2, A,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax1.plot(dl2, Aq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax1.plot(dl2, Ag, ':',  linewidth=2.6, color='tab:orange', label=r'Field')
    #
    ax2.plot(dl2, D,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax2.plot(dl2, Dq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax2.plot(dl2, Dg, ':',  linewidth=2.6, color='tab:orange', label=r'Field')
    #
    ax3.plot(dl2, c,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax3.plot(dl2, cq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax3.plot(dl2, cg, ':',  linewidth=2.6, color='tab:orange', label=r'Field')
    #
    ax1.set_ylabel(r'$A(\varDelta^2)$')
    ax2.set_ylabel(r'$D(\varDelta^2)$')
    ax3.set_ylabel(r'$\bar{c}(\varDelta^2)$')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
        ax.set_xscale('log')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    fig.patch.set_alpha(0)
    fig.savefig('yukawa_emtff.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check that "hydrogen" (muonium) wave function behaves reasonably

def hydrogen_check():
    dl2 = np.geomspace(1e-12, 0.1, 666)
    dl = np.sqrt(dl2)
    H = dwf_hydrogen()
    # One-body form factors
    Aq = emtff.AU(dl, wf=H, nff='point', impulse=True)
    Dq = emtff.DU(dl, wf=H, nff='point', impulse=True)
    cq = emtff.cU(dl, wf=H, nff='point', impulse=True)
    # Coulomb form factors
    field = {
            'g1' : -np.sqrt(4*np.pi*H.alpha),
            'g2' : np.sqrt(4*np.pi*H.alpha),
            'mf' : 0,
            's'  : 1
            }
    Ag = emtff.AU(dl, wf=H, nff='point', impulse=False, field=field)
    Dg = emtff.DU(dl, wf=H, nff='point', impulse=False, field=field)
    cg = emtff.cU(dl, wf=H, nff='point', impulse=False, field=field)
    A = Aq + Ag
    D = Dq + Dg
    c = cq + cg
    # Plot
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    cf = 1e6
    #
    ax1.plot(cf*dl2, A,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax1.plot(cf*dl2, Aq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax1.plot(cf*dl2, Ag, ':',  linewidth=2.6, color='tab:orange', label=r'Coulomb')
    #
    ax2.plot(cf*dl2, D,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax2.plot(cf*dl2, Dq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax2.plot(cf*dl2, Dg, ':',  linewidth=2.6, color='tab:orange', label=r'Coulomb')
    #
    ax3.plot(cf*dl2, c,  '-',  linewidth=2.6, color='black',      label=r'Total')
    ax3.plot(cf*dl2, cq, '--', linewidth=2.6, color='tab:blue',   label=r'Particle')
    ax3.plot(cf*dl2, cg, ':',  linewidth=2.6, color='tab:orange', label=r'Coulomb')
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
# Auxiliary function test

def auxtest():
    zetamax = 10
    zeta = np.linspace(1e-6, zetamax, 666)
    #
    zetamax_1 = 36
    zeta_1 = np.linspace(1e-6, zetamax_1, 666)
    delta_1 = 0
    omega_1 = 0
    anl_1 = emtff.yukawa.Phi_analytic(zeta_1, omega_1, delta_1)
    num_1 = emtff.yukawa.Phi_numeric( zeta_1, omega_1, delta_1)
    #
    zetamax_2 = 4
    zeta_2 = np.linspace(1e-6, zetamax_2, 666)
    delta_2 = 0.2
    omega_2 = 5
    anl_2 = emtff.yukawa.Phi_analytic(zeta_2, omega_2, delta_2)
    num_2 = emtff.yukawa.Phi_numeric( zeta_2, omega_2, delta_2)
    #
    zetamax_3 = 10
    zeta_3 = np.linspace(1e-6, zetamax_3, 666)
    delta_3 = 0.7
    omega_3 = 0.1
    anl_3 = emtff.yukawa.Phi_analytic(zeta_3, omega_3, delta_3)
    num_3 = emtff.yukawa.Phi_numeric( zeta_3, omega_3, delta_3)
    #
    nrows, ncols = 1, 3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    #
    ax1.plot(zeta_1, zeta_1*anl_1, '-',  linewidth=3, color='tab:orange', label=r'Analytic result')
    ax1.plot(zeta_1, zeta_1*num_1, '--', linewidth=3, color='tab:blue',   label=r'Numerical integral')
    ax1.plot(zeta_1, zeta_1*0,     '-',  linewidth=1, color='tab:gray')
    #
    ax2.plot(zeta_2, zeta_2*anl_2, '-',  linewidth=3, color='tab:orange', label=r'Analytic result')
    ax2.plot(zeta_2, zeta_2*num_2, '--', linewidth=3, color='tab:blue',   label=r'Numerical integral')
    ax2.plot(zeta_2, zeta_2*0,     '-',  linewidth=1, color='tab:gray')
    #
    ax3.plot(zeta_3, zeta_3*anl_3, '-',  linewidth=3, color='tab:orange', label=r'Analytic result')
    ax3.plot(zeta_3, zeta_3*num_3, '--', linewidth=3, color='tab:blue',   label=r'Numerical integral')
    ax3.plot(zeta_3, zeta_3*0,     '-',  linewidth=1, color='tab:gray')
    #
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(r'$\zeta$')
    eps = 0.03
    ax1.set_xlim((0-eps,zetamax_1+eps))
    ax2.set_xlim((0-eps,zetamax_2+eps))
    ax3.set_xlim((0-eps,zetamax_3+eps))
    ax1.set_ylabel(r'$\zeta \, \Phi(\zeta,\omega,\delta)$')
    legend = ax1.legend(prop = { 'size' : 26 }, loc=1)
    legend.get_frame().set_facecolor('#f8f8f8')
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='gray', boxstyle='round,pad=0.2')
    ax1.annotate(
            r'$\omega=0$, $\delta=0$', xy=(0.65,0.07), xycoords='axes fraction',
            bbox=bbox
            )
    ax2.annotate(
            r'$\omega=2$, $\delta=0.2$', xy=(0.61,0.89), xycoords='axes fraction',
            bbox=bbox
            )
    ax3.annotate(
            r'$\omega=0.1$, $\delta=0.7$', xy=(0.56,0.89), xycoords='axes fraction',
            bbox=bbox
            )
    #
    fig.patch.set_alpha(0)
    fig.savefig('auxtest.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Forward limit check

def _D0_intd(r, dwf):
    intd = 1/5 - 1/45*dwf.mu*r/hbar
    intd *= r * np.exp(-dwf.mu*r/hbar)
    intd += - 2/3*(1-np.exp(-dwf.mu*r/hbar))/(dwf.mu/hbar)
    intd *= dwf.u(r)**2
    intd *= -2*dwf.mNfm * dwf.alpha
    return intd

def _c0_intd(r, dwf):
    intd = dwf.u(r)**2*(1+dwf.mu*r/hbar) * np.exp(-dwf.mu*r/hbar) / r
    intd *= dwf.alpha/3 / (2*dwf.mNfm)
    return intd

def forward_test(mu):
    # TODO: work the forward limit into emtff.yukawa,
    # since the latter is numerically unstable at small Delta
    H = vwf_yukawa(mu=mu)
    D0_numi = emtff.DU(1e-3, wf=H, nff='point', impulse=False, yukawa=True)
    c0_numi = emtff.cU(0, wf=H, nff='point', impulse=False, yukawa=True)
    D0_true = quad(_D0_intd, 0, np.inf, args=(H,))[0]
    c0_true = quad(_c0_intd, 0, np.inf, args=(H,))[0]
    #
    D0_true += -2 * 2/3 * 2*H.mN / H.mu * H.alpha
    print("D0 via EMTFF method", D0_numi)
    print("D0 via analytic formula", D0_true)
    print("c0 via EMTFF method", c0_numi)
    print("c0 via analytic formula", c0_true)
    return


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.special import exp1 # for E1 test

from .. import emtff
from ..constants import hbar
from ..wf.airy import dwf_airy
from ..wf.hydrogen import dwf_hydrogen
from ..wf.variational import var_wf_airy, var_wf_harmonic, var_wf_cornell

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
    # Airy
    wf_exact_a = dwf_airy()
    wf_appr1_a = var_wf_airy(N=1)
    wf_appr2_a = var_wf_airy(N=2)
    wf_appr3_a = var_wf_airy(N=3)
    uX_a = wf_exact_a.u(r)
    u1_a = wf_appr1_a.u(r)
    u2_a = wf_appr2_a.u(r)
    u3_a = wf_appr3_a.u(r)
    # Harmonic oscillator
    omega0 = 1 # fm**-1
    mu = 1 # fm**-1
    wf_appr1_h = var_wf_harmonic(N=1, mN=2*mu*hbar, omega0=omega0)
    wf_appr2_h = var_wf_harmonic(N=2, mN=2*mu*hbar, omega0=omega0)
    wf_appr3_h = var_wf_harmonic(N=3, mN=2*mu*hbar, omega0=omega0)
    uX_h = 2*((omega0*mu)/np.pi)**(1/4) * r * np.exp(-omega0*mu*r**2/2)
    u1_h = wf_appr1_h.u(r)
    u2_h = wf_appr2_h.u(r)
    u3_h = wf_appr3_h.u(r)
    # Cornell potential
    wf_appr1_c = var_wf_cornell(N=1)
    wf_appr2_c = var_wf_cornell(N=2)
    wf_appr3_c = var_wf_cornell(N=3)
    wf_appr4_c = var_wf_cornell(N=4)
    wf_appr5_c = var_wf_cornell(N=5)
    wf_appr6_c = var_wf_cornell(N=6)
    wf_appr7_c = var_wf_cornell(N=7)
    wf_appr8_c = var_wf_cornell(N=8)
    wf_appr9_c = var_wf_cornell(N=9)
    u1_c = wf_appr1_c.u(r)
    u2_c = wf_appr2_c.u(r)
    u3_c = wf_appr3_c.u(r)
    u4_c = wf_appr4_c.u(r)
    u5_c = wf_appr5_c.u(r)
    u6_c = wf_appr6_c.u(r)
    u7_c = wf_appr7_c.u(r)
    u8_c = wf_appr8_c.u(r)
    u9_c = wf_appr9_c.u(r)
    # Plots
    nrows,ncols=3,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    ax4 = plt.subplot(nrows,ncols,4)
    ax5 = plt.subplot(nrows,ncols,5)
    ax6 = plt.subplot(nrows,ncols,6)
    ax7 = plt.subplot(nrows,ncols,7)
    ax8 = plt.subplot(nrows,ncols,8)
    ax9 = plt.subplot(nrows,ncols,9)
    #
    ax1.plot(r, uX_a, '-',  color='black',      linewidth=2.6, label=r'Exact')
    ax1.plot(r, u1_a, ':',  color='tab:blue',   linewidth=2.6, label=r'$N=1$')
    ax1.plot(r, u2_a, '-.', color='tab:orange', linewidth=2.6, label=r'$N=2$')
    ax1.plot(r, u3_a, '--', color='tab:green',  linewidth=2.6, label=r'$N=3$')
    #
    ax2.plot(r, uX_h, '-',  color='black',      linewidth=2.6)
    ax2.plot(r, u1_h, ':',  color='tab:blue',   linewidth=2.6)
    ax2.plot(r, u2_h, '-.', color='tab:orange', linewidth=2.6)
    ax2.plot(r, u3_h, '--', color='tab:green',  linewidth=2.6)
    #
    ax3.plot(r, u1_c, ':',  color='tab:blue',   linewidth=2.6)
    ax3.plot(r, u2_c, '-.', color='tab:orange', linewidth=2.6)
    ax3.plot(r, u3_c, '--', color='tab:green',  linewidth=2.6)
    ax3.plot(r, u4_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u5_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u6_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u7_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u7_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u8_c, '-',  color='tab:gray',   linewidth=1)
    ax3.plot(r, u9_c, '-',  color='tab:gray',   linewidth=1)
    #
    ax4.plot(r, u2_a-u1_a, '-.', color='tab:orange', linewidth=2.6)
    ax4.plot(r, u3_a-u2_a, '--', color='tab:green',  linewidth=2.6)
    #
    ax5.plot(r, u2_h-u1_h, '-.', color='tab:orange', linewidth=2.6)
    ax5.plot(r, u3_h-u2_h, '--', color='tab:green',  linewidth=2.6)
    #
    ax6.plot(r, u2_c-u1_c, ':',  color='tab:blue',   linewidth=2.6)
    ax6.plot(r, u3_c-u2_c, '-.', color='tab:orange', linewidth=2.6)
    ax6.plot(r, u4_c-u3_c, '--', color='tab:green',  linewidth=2.6)
    ax6.plot(r, u5_c-u4_c, '-',  color='tab:gray',   linewidth=1)
    ax6.plot(r, u6_c-u5_c, '-',  color='tab:gray',   linewidth=1)
    ax6.plot(r, u7_c-u6_c, '-',  color='tab:gray',   linewidth=1)
    ax6.plot(r, u7_c-u6_c, ':',  color='tab:gray',   linewidth=1)
    ax6.plot(r, u8_c-u7_c, '-',  color='tab:gray',   linewidth=1)
    ax6.plot(r, u9_c-u8_c, '-',  color='tab:gray',   linewidth=1)
    #
    ax7.plot(np.array([1,3]), wf_exact_a.Efm*hbar*np.ones(2), '-', color='black')
    ax7.plot(1, wf_appr1_a.E, 'o', color='tab:blue')
    ax7.plot(2, wf_appr2_a.E, 'o', color='tab:orange')
    ax7.plot(3, wf_appr3_a.E, 'o', color='tab:green')
    #
    ax8.plot(np.array([1,3]), 3/2*omega0*hbar*np.ones(2), '-', color='black')
    ax8.plot(1, wf_appr1_h.E, 'o', color='tab:blue')
    ax8.plot(2, wf_appr2_h.E, 'o', color='tab:orange')
    ax8.plot(3, wf_appr3_h.E, 'o', color='tab:green')
    #
    ax9.plot(1, wf_appr1_c.E, 'o', color='tab:blue')
    ax9.plot(2, wf_appr2_c.E, 'o', color='tab:orange')
    ax9.plot(3, wf_appr3_c.E, 'o', color='tab:green')
    ax9.plot(4, wf_appr4_c.E, 'o', color='tab:gray')
    ax9.plot(5, wf_appr5_c.E, 'o', color='tab:gray')
    ax9.plot(6, wf_appr6_c.E, 'o', color='tab:gray')
    ax9.plot(7, wf_appr7_c.E, 'o', color='tab:gray')
    ax9.plot(8, wf_appr8_c.E, 'o', color='tab:gray')
    ax9.plot(9, wf_appr9_c.E, 'o', color='tab:gray')
    #
    ax1.set_ylabel(r'$u(r)$ (fm$^{-1/2}$)')
    ax4.set_ylabel(r'$u_N(r) - u_{N-1}(r)$ (fm$^{-1/2}$)')
    ax7.set_ylabel(r'$E$ (GeV)')
    for ax in [ax1]:#, ax2, ax3]:
        ax.get_xaxis().set_visible(False)
        l = ax.legend(prop = { 'size' : 27 }, loc=1)
        l.get_frame().set_facecolor('#f8f8f8')
    #for ax in [ax2, ax3, ax5, ax6, ax8, ax9]:
    #    ax.get_yaxis().set_visible(False)
    for ax in [ax2, ax3, ax5, ax6]:
        ax.get_yaxis().set_visible(False)
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim((-0.1, 1.3))
    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel(r'$r$ (fm)')
        ax.set_ylim((-0.063, 0.063))
    for ax in [ax7, ax8, ax9]:
        ax.set_xlabel(r'$N$')
        #ax.set_ylim((0, 1))
    fig.patch.set_alpha(0)
    fig.savefig('variational_test.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check that Cornell wave function behaves reasonably

def cornell_check(N=4):
    dl2 = np.linspace(1e-4, 10, 666)
    dl = np.sqrt(dl2)
    eta_c = var_wf_cornell(N=N)
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

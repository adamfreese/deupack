import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.special import exp1 # for E1 test
from scipy.integrate import quad

from .. import emtff
from ..constants import hbar, alphaQED, GN, m_kep
from ..wf.airy import dwf_airy
from ..wf.hydrogen import dwf_hydrogen
from ..wf.variational import vwf_cornell, vwf_yukawa, vwf_multifield

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
    # Fixed parameters
    N = 1
    m = 1
    mu = 0.1
    alpha = 1
    Nmax = 4
    # Separation variable
    r = np.linspace(0, 4, 666)
    # Lines and colors
    lines = [':', '-.', '--', '-']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    # Wave function lists
    wf = []
    u = []
    E = []
    a = []
    k = []
    R = []
    for n in range(Nmax):
        wf += [ vwf_multifield(N=n+1, alpha=alpha, mN=m, mu=mu) ]
        E  += [ wf[n].E ]
        a  += [ wf[n].a ]
        k  += [ wf[n].kfm*hbar ]
        u  += [ wf[n].u(r) ]
        R  += [ np.sqrt(-E[n])/k[n] ]
    # Print out data for table
    for n in range(Nmax):
        print("*"*80)
        print("N={:d}".format(n+1))
        print("Energy:", E[n])
        print("Decay:", k[n])
        print("Ratio:", R[n])
        print("Coefficients:", a[n])
    # Plots
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    # Plot wave functions and energy estimates
    for n in range(Nmax):
        ax1.plot(r, u[n], lines[n], linewidth=2.6,
                 zorder = Nmax-n,
                 color = colors[n],
                 label=r'$N='+"{:d}".format(n+1) + r'$')
        ax2.plot(n+1, E[n], 'o', color=colors[n])
    # Labels etc
    ax1.set_ylabel(r'$u(r)$ (fm$^{-1/2}$)')
    ax1.set_xlabel(r'$r$ (fm)')
    ax2.set_ylabel(r'$E$ (GeV)')
    ax2.set_xlabel(r'$N$')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    l.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('variational_test.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some specific three-panel plots

def mff_muonium():
    dl2 = np.geomspace(1e-12, 0.1, 666)
    wf = dwf_hydrogen(n=1, l=0, ml=0)
    field = {
            'g1' : -np.sqrt(4*np.pi*alphaQED),
            'g2' :  np.sqrt(4*np.pi*alphaQED),
            'mu' : 0,
            's'  : 1
            }
    fig = _mff_3panel(wf, field, dl2, units='MeV')
    fig.savefig('mff_muonium.pdf')
    return

def mff_keplerium():
    dl2 = np.geomspace(1e-12, 0.1, 666)
    wf = dwf_hydrogen(n=1, l=0, ml=0, mN=m_kep, alpha=GN*m_kep**2)
    field = {
            'g1' : np.sqrt(4*np.pi*GN)*m_kep,
            'g2' : np.sqrt(4*np.pi*GN)*m_kep,
            'mu' : 0,
            's'  : 2
            }
    fig = _mff_3panel(wf, field, dl2, units='MeV')
    fig.savefig('mff_keplerium.pdf')
    return

def mff_yukawa_zero():
    dl2 = np.geomspace(1e-4, 1000, 666)
    wf = vwf_yukawa(N=3, mu=0.1, alpha=1, mN=1)
    field = {
            'g1' : np.sqrt(4*np.pi),
            'g2' : np.sqrt(4*np.pi),
            'mu' : 0.1,
            's'  : 0
            }
    fig = _mff_3panel(wf, field, dl2, units='GeV')
    fig.savefig('mff_yukawa_zero.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Yukawa two-panel D comparison

def mff_yukawa_D():
    # Field mass values
    N_mu = 4
    muse = [0.01, 0.04, 0.07, 0.1]
    # Lines and colors
    lines = [':', '-.', '--', '-']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    # Fixed parameters
    m = 1
    alpha = 1
    g = np.sqrt(4*np.pi*alpha)
    # Momentum transfer array
    dl2 = np.geomspace(1e-4, 1000, 666)
    dl = np.sqrt(dl2)
    # Arrays of D form factors
    D0 = []
    D1 = []
    # Create the fields, wave functions and D-terms
    for n in range(N_mu):
        wf = vwf_yukawa(N=3, alpha=alpha, mN=m, mu=muse[n])
        D0 += [
                emtff.DU(dl, wf=wf, nff='point', impulse=True,
                         field={ 'g1': g, 'g2': g, 'mu': muse[n], 's': 0}
                         )
                ]
        D1 += [
                emtff.DU(dl, wf=wf, nff='point', impulse=True,
                         field={ 'g1': g, 'g2': -g, 'mu': muse[n], 's': 1}
                         )
                ]
    # Set up canvas
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax0 = plt.subplot(nrows,ncols,1)
    ax1 = plt.subplot(nrows,ncols,2)
    # Plot D form factors
    for n in range(N_mu):
        ax0.plot(dl2, D0[n], lines[n], linewidth=2.6,
                 color = colors[n],
                 zorder = N_mu-n,
                 label = r'$\mu = '+'{:d}'.format(int(1000*muse[n]))+r'$~MeV'
                 )
        ax1.plot(dl2, D1[n], lines[n], linewidth=2.6,
                 color = colors[n],
                 zorder = N_mu-n,
                 label = r'$\mu = '+'{:d}'.format(int(1000*muse[n]))+r'$~MeV'
                 )
    # Finish up plot
    for ax in [ax0,ax1]:
        ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
        ax.set_xscale('log')
        ax.set_ylabel(r'$D(\varDelta^2)$')
    l = ax0.legend(prop = { 'size' : 27 }, loc=4)
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='darkgray', boxstyle='round,pad=0.2')
    ax0.annotate(
            r'\textbf{Spin-zero}', xy=(0.04,0.067), xycoords='axes fraction',
            bbox=bbox
            )
    ax1.annotate(
            r'\textbf{Spin-one}', xy=(0.04,0.067), xycoords='axes fraction',
            bbox=bbox
            )
    l.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('mff_yukawa_D.pdf')
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Multifield D-term plot

def mff_multifield():
    # TODO
    # Fixed parameters
    alpha = 1
    mu = 0.1
    m = 1
    # Set up wave function and fields
    # Prepare canvas
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    # Momentum transfer ranges
    dl2_a = np.geomspace(1e-6,  1e3, 666)
    dl2_b = np.geomspace(1e-78, 1e-69, 666)
    # Create panels
    _multifield_panel(ax1, dl2_a, sign= 1, alpha=alpha, m=m, mu=mu)
    _multifield_panel(ax2, dl2_a, sign=-1, alpha=alpha, m=m, mu=mu)
    _multifield_panel(ax3, dl2_b, sign=-1, alpha=alpha, m=m, mu=mu)
    # Finish up
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    l.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    fig.savefig('mff_multifield.pdf')
    return

def _multifield_panel(ax, dl2,
                      sign = 1,
                      m = 1, mu = 0.1, alpha = 1
                      ):
    # Set up wave function
    wf = vwf_multifield(N=3, mu=mu, alpha=alpha, mN=m, sign=1)
    g = np.sqrt(4*np.pi*alpha)
    e = np.sqrt(4*np.pi*alphaQED)
    gG = np.sqrt(4*np.pi*GN) * m
    # Set up fields
    # Also, set up a split points because I'm going to consider really small
    # momenta transfer at which the exact formulas become numerically unstable.
    field_yk = { 'g1' : g,  'g2' : g,      'mu' : mu, 's'  : 0, 'k0' : 1e-4 }
    field_em = { 'g1' : e,  'g2' : sign*e, 'mu' : 0,  's'  : 1, 'k0' : 1e-6 }
    field_gr = { 'g1' : gG, 'g2' : gG,     'mu' : 0,  's'  : 2, 'k0' : 1e-20 }
    # Get form factors
    dl = np.sqrt(dl2)
    Dn = emtff.DU(dl, wf=wf, nff='point', impulse=True)
    Dy = emtff.DU(dl, wf=wf, nff='point', impulse=False, field=field_yk)
    Dc = emtff.DU(dl, wf=wf, nff='point', impulse=False, field=field_em)
    Dg = emtff.DU(dl, wf=wf, nff='point', impulse=False, field=field_gr)
    D = Dn + Dy + Dc + Dg
    # Plot the form factors
    ax.plot(dl2, D,  '-',  linewidth=2.6, color='black',      zorder=1, label=r'Total')
    ax.plot(dl2, Dn, '-',  linewidth=2.6, color='tab:blue',   zorder=2, label=r'Particle')
    ax.plot(dl2, Dy, '--', linewidth=2.6, color='tab:orange', zorder=3, label=r'Yukawa')
    ax.plot(dl2, Dc, '-.', linewidth=2.6, color='tab:green',  zorder=4, label=r'Coulomb')
    ax.plot(dl2, Dg, ':',  linewidth=2.6, color='tab:purple', zorder=5, label=r'Gravity')
    # Axis labels
    ax.set_ylabel(r'$D(\varDelta^2)$')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_xscale('log')
    # Annotation for sign
    if(sign==1):
        text = r'\textbf{Equal charges}'
    if(sign==-1):
        text = r'\textbf{Opposite charges}'
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='darkgray', boxstyle='round,pad=0.2')
    ax.annotate(
            text, xy=(0.957,0.067), xycoords='axes fraction',
            horizontalalignment='right',
            bbox=bbox
            )
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
    anl_1 = emtff.abelian.Phi_analytic(zeta_1, omega_1, delta_1)
    num_1 = emtff.abelian.Phi_numeric( zeta_1, omega_1, delta_1)
    #
    zetamax_2 = 4
    zeta_2 = np.linspace(1e-6, zetamax_2, 666)
    delta_2 = 0.2
    omega_2 = 5
    anl_2 = emtff.abelian.Phi_analytic(zeta_2, omega_2, delta_2)
    num_2 = emtff.abelian.Phi_numeric( zeta_2, omega_2, delta_2)
    #
    zetamax_3 = 10
    zeta_3 = np.linspace(1e-6, zetamax_3, 666)
    delta_3 = 0.7
    omega_3 = 0.1
    anl_3 = emtff.abelian.Phi_analytic(zeta_3, omega_3, delta_3)
    num_3 = emtff.abelian.Phi_numeric( zeta_3, omega_3, delta_3)
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
    bbox = dict(facecolor='#f8f8f8', alpha=0.76, edgecolor='darkgray', boxstyle='round,pad=0.2')
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
# Under the hood stuff for three-panel MFF plots

def _mff_3panel(wf, field,
                dl2,
                units = 'GeV',
                ):
    # Deal with unit conversions if needed
    cf = 1
    if(units=='MeV'):
        cf = 1e6
    # Form factors
    dl = np.sqrt(dl2)
    Aq = emtff.AU(dl, wf=wf, nff='point', impulse=True)
    Dq = emtff.DU(dl, wf=wf, nff='point', impulse=True)
    cq = emtff.cU(dl, wf=wf, nff='point', impulse=True)
    Ag = emtff.AU(dl, wf=wf, nff='point', impulse=False, field=field)
    Dg = emtff.DU(dl, wf=wf, nff='point', impulse=False, field=field)
    cg = emtff.cU(dl, wf=wf, nff='point', impulse=False, field=field)
    A = Aq + Ag
    D = Dq + Dg
    c = cq + cg
    # Set up canvas
    nrows,ncols=1,3
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax1 = plt.subplot(nrows,ncols,1)
    ax2 = plt.subplot(nrows,ncols,2)
    ax3 = plt.subplot(nrows,ncols,3)
    # Plot the form factors
    _mff_single_panel(ax1, cf*dl2, A, Aq, Ag)
    _mff_single_panel(ax2, cf*dl2, D, Dq, Dg)
    _mff_single_panel(ax3, cf*dl2, c, cq, cg)
    # Finish up plot
    ax1.set_ylabel(r'$A(\varDelta^2)$')
    ax2.set_ylabel(r'$D(\varDelta^2)$')
    ax3.set_ylabel(r'$\bar{c}(\varDelta^2)$')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(r'$\varDelta^2$ ('+units+r'$^2$)')
        ax.set_xscale('log')
    l = ax1.legend(prop = { 'size' : 27 }, loc=1)
    l.get_frame().set_facecolor('#f8f8f8')
    fig.patch.set_alpha(0)
    return fig

def _mff_single_panel(ax, dl2, F, Fq, Fg):
    ax.plot(dl2, F,  '-',  linewidth=2.6, color='black',     zorder=1, label=r'Total')
    ax.plot(dl2, Fq, '--', linewidth=2.6, color='tab:blue',  zorder=2, label=r'Particle')
    ax.plot(dl2, Fg, '-.', linewidth=2.6, color='tab:orange',zorder=3, label=r'Field')
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
    # TODO: work the forward limit into emtff.abelian,
    # since the latter is numerically unstable at small Delta
    H = vwf_yukawa(mu=mu)
    field = {
            'g1' : np.sqrt(4*np.pi*H.alpha),
            'g2' : np.sqrt(4*np.pi*H.alpha),
            'mu' : H.mu,
            's'  : 0
            }
    D0_numi = emtff.DU(1e-3, wf=H, nff='point', impulse=False, field=field)
    c0_numi = emtff.cU(0,    wf=H, nff='point', impulse=False, field=field)
    D0_true = quad(_D0_intd, 0, np.inf, args=(H,))[0]
    c0_true = quad(_c0_intd, 0, np.inf, args=(H,))[0]
    #
    D0_true += -2 * 2/3 * 2*H.mN / H.mu * H.alpha
    print("D0 via EMTFF method", D0_numi)
    print("D0 via analytic formula", D0_true)
    print("c0 via EMTFF method", c0_numi)
    print("c0 via analytic formula", c0_true)
    return


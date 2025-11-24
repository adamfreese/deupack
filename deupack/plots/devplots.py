import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from .. import mff
from ..density import *
from ..mff.nucleon.chooser import choose_nff

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
    nrows,ncols=2,5
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
    l = ax_AU.legend(prop = { 'size' : 27 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('nff_comparisons.pdf')
    return

def plot_wf_comparisons():
    nrows,ncols=2,5
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
# Formula comparison for cT2

def plot_cT2():
    dl2 = np.geomspace(1e-6, 1e1, 666)
    cT2_Adam  = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2')
    cT2_Alan  = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Alan')
    cT2_Adam3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Adam3')
    cT2_Alan3 = mff.cT2(np.sqrt(dl2), nff='point', formula='cT2Alan3')
    #
    nrows,ncols=1,1
    fig = plt.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = plt.subplot(nrows,ncols,1)
    #
    ax.plot(dl2, cT2_Adam,  '-',  linewidth=2, color='xkcd:true green', label=r'Adam')
    ax.plot(dl2, cT2_Alan,  '--', linewidth=2, color='xkcd:rich purple',label=r'Alan')
    ax.plot(dl2, cT2_Adam3, '-.', linewidth=2, color='xkcd:cobalt',     label=r'Adam3')
    ax.plot(dl2, cT2_Alan3, ':',  linewidth=2, color='xkcd:coral',      label=r'Alan3')
    #
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$\bar{c}_{T2}(\varDelta^2)$')
    l = ax.legend(prop = { 'size' : 27 }, loc=4)
    fig.patch.set_alpha(0)
    fig.savefig('cT2.pdf')
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
# 3D density plots

def plot_mass_3d(nff='ba', wf='av18',
                 nb=150, # try smaller number (e.g., 50) for testing
                 bmax=2.8):
    t_0 = time.time()
    D = Density(nff=nff, wf=wf)
    b = np.linspace(-bmax, bmax, nb)
    MU = D.mass_3D_U(b,b,b)
    MT = D.mass_3D_T(b,b,b)
    M0 = MU + 2/3*MT
    M1 = MU - 1/3*MT
    t_A = time.time()
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*11,nrows*11))
    labels = [r'$m_j=0$', r'$m_j=1$']
    multidensity3d(fig, b, b, b,
                   nrows, ncols,
                   M0, M1,
                   labels=labels,
                   clabel=r'Two-dimensional mass densities (GeV/fm$^2$)',
                   decay=3, opacity=0.53, cmap=cmr.voltage,
                   projections=True, divergent=False, s=1)
    t_B = time.time()
    fig.savefig('mass3D_{}_{}.png'.format(wf,nff), dpi=150)
    t_C = time.time()
    print("Time to calculate mass densities: {:.3f} s".format(t_A-t_0))
    print("Time to plot mass densities:      {:.3f} s".format(t_B-t_A))
    print("Time to save plots:               {:.3f} s".format(t_C-t_B))
    return

def plot_stress_3d(nff='ba', wf='av18',
                   nb=150, # try smaller number (e.g., 50) for testing
                   bmax=2.8):
    t_0 = time.time()
    # Load or create the stresses
    b = np.linspace(-bmax, bmax, nb)
    pr0,pt0,pz0,pr1,pt1,pz1 = get_stresses(nff=nff, wf=wf, nb=nb, bmax=bmax)
    t_A = time.time()
    # Make some labels
    labels = [
            r'$m_j=0$, radial pressure',
            r'$m_j=0$, azimuthal pressure',
            r'$m_j=0$, $z$-direction pressure',
            r'$m_j=1$, radial pressure',
            r'$m_j=1$, azimuthal pressure',
            r'$m_j=1$, $z$-direction pressure'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    multidensity3d(fig, b, b, b,
                   nrows, ncols,
                   pr0, pt0, pz0, pr1, pt1, pz1,
                   labels=labels,
                   clabel=r'Two-dimensional pressure projections (GeV/fm$^2$)',
                   decay=3, opacity=0.53, cmap=cmr.iceburn,
                   projections=True, divergent=True, s=1)
    t_B = time.time()
    ### Save as png, because pdf is insanely large
    fig.savefig('stress3D_{}_{}.png'.format(wf,nff), dpi=150)
    t_C = time.time()
    print("Time to calculate (or load) pressures: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:                {:.3f} s".format(t_B-t_A))
    print("Time to save plots:                    {:.3f} s".format(t_C-t_B))
    return

def get_stresses(nff='ba', wf='av18', nb=150, bmax=2.8):
    # Try loading a cache file.
    # If that doesn't work, then we actually need to make the densities.
    # But if we need to make them, save to a cache since this is expensive.
    cachefile = "stress_{}_{}_{:d}_{:.2f}.npy".format(nff,wf,nb,bmax)
    try:
        data = np.load(cachefile)
        pr0,pt0,pz0,pr1,pt1,pz1 = data
    except:
        D = Density(nff=nff, wf=wf)
        b = np.linspace(-bmax, bmax, nb)
        # Get the full stress tensor first
        TijU   = D.stress_3D_U(b,b,b)
        TijT   = D.stress_3D_T(b,b,b)
        # Get the r, phi and z projections
        rhat   = make_rhat(  b,b,b)
        phihat = make_phihat(b,b,b)
        zhat   = make_zhat(  b,b,b)
        prU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, rhat, rhat)
        ptU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, phihat, phihat)
        pzU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, zhat, zhat)
        prT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, rhat, rhat)
        ptT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, phihat, phihat)
        pzT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, zhat, zhat)
        # Get the spin projections
        pr0 = prU + 2/3*prT
        pt0 = ptU + 2/3*ptT
        pz0 = pzU + 2/3*pzT
        pr1 = prU - 1/3*prT
        pt1 = ptU - 1/3*ptT
        pz1 = pzU - 1/3*pzT
        # cache this since it's expensive
        np.save(cachefile, [pr0,pt0,pz0,pr1,pt1,pz1])
    return pr0,pt0,pz0,pr1,pt1,pz1

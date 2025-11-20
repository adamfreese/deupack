import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as py

from .. import mff
from ..density import *
from ..mff.nucleon.chooser import choose_nff

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
py.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Routines to make specific plots

def plot_nff_comparisons():
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

def plot_group_comparisons():
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
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
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
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
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
# Pixel plot to test densities

def density_test():
    print("Flag A")
    D = Density()
    print("Flag B")
    b = np.linspace(-2, 2, 100)
    # Slice at y=0
    MU_y0 = D.mass_3D_U(b,0,b)[:,0,:]
    MT_y0 = D.mass_3D_T(b,0,b)[:,0,:]
    M0_y0 = MU_y0 + 2/3*MT_y0
    M1_y0 = MU_y0 - 1/3*MT_y0
    # Slice at z=0
    MU_z0 = D.mass_3D_U(b,b,0)[:,:,0]
    MT_z0 = D.mass_3D_T(b,b,0)[:,:,0]
    M0_z0 = MU_z0 + 2/3*MT_z0
    M1_z0 = MU_z0 - 1/3*MT_z0
    #
    vmax = MU_y0.max() + MT_y0.max()
    #
    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = py.subplot(nrows,ncols,1, aspect='equal')
    ax2 = py.subplot(nrows,ncols,2, aspect='equal')
    ax3 = py.subplot(nrows,ncols,3, aspect='equal')
    ax4 = py.subplot(nrows,ncols,4, aspect='equal')
    c1 = ax1.pcolormesh(b, b, M0_z0.T, vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    c2 = ax2.pcolormesh(b, b, M1_z0.T, vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    c3 = ax3.pcolormesh(b, b, M0_y0  , vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    c4 = ax4.pcolormesh(b, b, M1_y0  , vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$x$ (fm)')
        ax.set_ylabel(r'$y$ (fm)')
    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$z$ (fm)')
        ax.set_ylabel(r'$x$ (fm)')
    ax1.set_title('$s= 0$, top view')
    ax2.set_title('$s=+1$, top view')
    ax3.set_title('$s= 0$, side view')
    ax4.set_title('$s=+1$, side view')
    #
    fig.patch.set_alpha(0)
    fig.savefig('mass.pdf')
    return

def density_test_2():
    print("Flag A")
    D = Density()
    print("Flag B")
    b = np.linspace(-2, 2, 100)
    # New scheme: use built-in radial pressure methods
    prU_z0 = D.pr_3D_U(b,b,0)[:,:,0,...]
    prU_y0 = D.pr_3D_U(b,0,b)[:,0,:,...]
    prT_z0 = D.pr_3D_T(b,b,0)[:,:,0,...]
    prT_y0 = D.pr_3D_T(b,0,b)[:,0,:,...]
    # Pure states
    pr1_z0 = prU_z0 - 1/3*prT_z0
    pr1_y0 = prU_y0 - 1/3*prT_y0
    pr0_z0 = prU_z0 + 2/3*prT_z0
    pr0_y0 = prU_y0 + 2/3*prT_y0
    print("Flag C")
    print(pr0_y0.min(), pr0_z0.min(), pr1_y0.min(), pr1_z0.min())
    #
    vmax = max(abs(pr0_z0).max(), abs(pr0_y0).max(), abs(pr1_z0).max(), abs(pr1_y0).max(),)
    #
    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = py.subplot(nrows,ncols,1, aspect='equal')
    ax2 = py.subplot(nrows,ncols,2, aspect='equal')
    ax3 = py.subplot(nrows,ncols,3, aspect='equal')
    ax4 = py.subplot(nrows,ncols,4, aspect='equal')
    c1 = ax1.pcolormesh(b, b, pr0_z0.T, vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    c2 = ax2.pcolormesh(b, b, pr1_z0.T, vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    c3 = ax3.pcolormesh(b, b, pr0_y0  , vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    c4 = ax4.pcolormesh(b, b, pr1_y0  , vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$x$ (fm)')
        ax.set_ylabel(r'$y$ (fm)')
    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$z$ (fm)')
        ax.set_ylabel(r'$x$ (fm)')
    ax1.set_title('$s= 0$, top view')
    ax2.set_title('$s=+1$, top view')
    ax3.set_title('$s= 0$, side view')
    ax4.set_title('$s=+1$, side view')
    #
    fig.patch.set_alpha(0)
    fig.savefig('radial_pressure.pdf')
    return

def plot_stress_slice(zero_axis='z', s=0):
    D = Density()
    b = np.linspace(-2, 2, 100)
    if(zero_axis=='z'):
        x,y,z = b,b,0
    elif(zero_axis=='x'):
        x,y,z = 0,b,b
    elif(zero_axis=='y'):
        x,y,z = b,0,b
    else:
        raise ValueError("Invalid value for zero_azis: {}.".format(zero_axis))
    TijU = D.stress_3D_U(x,y,z)
    TijT = D.stress_3D_T(x,y,z)
    rhat = make_rhat(x,y,z)
    phihat = make_phihat(x,y,z)
    zhat = make_zhat(x,y,z)
    prU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, rhat, rhat)
    ptU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, phihat, phihat)
    pzU = np.einsum('xyzij,xyzi,xyzj->xyz', TijU, zhat, zhat)
    prT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, rhat, rhat)
    ptT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, phihat, phihat)
    pzT = np.einsum('xyzij,xyzi,xyzj->xyz', TijT, zhat, zhat)
    if(s==0):
        pr = prU + 2/3*prT
        pt = ptU + 2/3*ptT
        pz = pzU + 2/3*pzT
    elif(s==1 or s==-1):
        pr = prU - 1/3*prT
        pt = ptU - 1/3*ptT
        pz = pzU - 1/3*pzT
    else:
        raise ValueError("s={:d} is not a valid spin.".format(s))
    vmax = max( abs(pr).max(), abs(pt).max(), abs(pz).max() )
    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = py.subplot(nrows,ncols,1, aspect='equal')
    ax2 = py.subplot(nrows,ncols,2, aspect='equal')
    ax3 = py.subplot(nrows,ncols,3, aspect='equal')
    c1 = ax1.pcolormesh(b, b, np.squeeze(pr).T, vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    c2 = ax2.pcolormesh(b, b, np.squeeze(pt).T, vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    c3 = ax3.pcolormesh(b, b, np.squeeze(pz).T, vmin=-vmax, vmax=vmax, cmap='PRGn', shading='gouraud')
    ax1.set_title('Radial stress')
    ax2.set_title('Azimuthal stress')
    ax3.set_title('$z$-direction stress')
    #
    fig.patch.set_alpha(0)
    fig.savefig('stress.pdf')
    return


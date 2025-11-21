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

from .densityplot3d import densityplot3d

# Testing stuff
import time

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
# Density plots

def plot_mass_3d(nff='ba', wf='av18'):
    D = Density(nff=nff, wf=wf)
    bmax = 2.12
    nb = 180
    #nb = 60 # smaller to test changes quickly
    b = np.linspace(-bmax, bmax, nb)
    MU = D.mass_3D_U(b,b,b)
    MT = D.mass_3D_T(b,b,b)
    M0 = MU + 2/3*MT
    M1 = MU - 1/3*MT
    # Prepare figure
    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*11,nrows*11))
    ax1 = py.subplot(nrows,ncols,1,projection='3d')
    ax2 = py.subplot(nrows,ncols,2,projection='3d')
    for ax in [ax1, ax2]:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((0,0,0,1))
            axis.pane.set_edgecolor('gray')
        ax.grid(False)
        ax.patch.set_alpha(0)
        ax.set_xlabel('\n'+r'$x$ (fm)')
        ax.set_ylabel('\n'+r'$y$ (fm)')
        ax.set_zlabel('\n'+r'$z$ (fm)')
    # 3D densities
    densityplot3d(ax1, b, b, b, M0, s=2, opacity=0.26, decay=4, projections=True, cmap=py.cm.magma)
    densityplot3d(ax2, b, b, b, M1, s=2, opacity=0.26, decay=4, projections=True, cmap=py.cm.magma)
    # Some labels
    bbox = dict(facecolor='white', alpha=0.76, edgecolor='black', boxstyle='round,pad=0.5')
    ax1.text2D(0.05, 0.95, r'$m_j=0$', transform=ax1.transAxes, bbox=bbox)
    ax2.text2D(0.05, 0.95, r'$m_j=1$', transform=ax2.transAxes, bbox=bbox)
    # Save as png, because pdf is insanely large
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig('mass3D_{}_{}.png'.format(wf,nff), dpi=150)
    return

def plot_stress_3d(nff='ba', wf='av18',
                   nb=180, # try smaller number (e.g., 60) for testing
                   bmax=2.12):
    # TODO: maybe locally cache the densities,
    # since they take a long time to compute for large nb
    t_0 = time.time()
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
    t_A = time.time()
    # Prepare figure
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*11,nrows*11))
    ax0r = py.subplot(nrows,ncols,1,projection='3d')
    ax0t = py.subplot(nrows,ncols,2,projection='3d')
    ax0z = py.subplot(nrows,ncols,3,projection='3d')
    ax1r = py.subplot(nrows,ncols,4,projection='3d')
    ax1t = py.subplot(nrows,ncols,5,projection='3d')
    ax1z = py.subplot(nrows,ncols,6,projection='3d')
    for ax in [ax0r, ax0t, ax0z, ax1r, ax1t, ax1z]:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((0,0,0,1))
            axis.pane.set_edgecolor('gray')
        ax.grid(False)
        ax.patch.set_alpha(0)
        ax.set_xlabel('\n'+r'$x$ (fm)')
        ax.set_ylabel('\n'+r'$y$ (fm)')
        ax.set_zlabel('\n'+r'$z$ (fm)')
    # 3D densities
    kwargs = {
            's' : 2, 'opacity' : 0.26, 'decay' : 4, 'cmap' : py.cm.vanimo,
            'projections' : True, 'divergent' : True
            }
    densityplot3d(ax0r, b, b, b, pr0, **kwargs)
    densityplot3d(ax0t, b, b, b, pt0, **kwargs)
    densityplot3d(ax0z, b, b, b, pz0, **kwargs)
    densityplot3d(ax1r, b, b, b, pr1, **kwargs)
    densityplot3d(ax1t, b, b, b, pt1, **kwargs)
    densityplot3d(ax1z, b, b, b, pz1, **kwargs)
    t_B = time.time()
    # Some labels
    bbox = dict(facecolor='white', alpha=0.76, edgecolor='black', boxstyle='round,pad=0.5')
    ax0r.text2D(0.05, 0.95, r'$m_j=0$, radial pressure',        transform=ax0r.transAxes, bbox=bbox)
    ax1r.text2D(0.05, 0.95, r'$m_j=1$, radial pressure',        transform=ax1r.transAxes, bbox=bbox)
    ax0t.text2D(0.05, 0.95, r'$m_j=0$, azimuthal pressure',     transform=ax0t.transAxes, bbox=bbox)
    ax1t.text2D(0.05, 0.95, r'$m_j=1$, azimuthal pressure',     transform=ax1t.transAxes, bbox=bbox)
    ax0z.text2D(0.05, 0.95, r'$m_j=0$, $z$-direction pressure', transform=ax0z.transAxes, bbox=bbox)
    ax1z.text2D(0.05, 0.95, r'$m_j=1$, $z$-direction pressure', transform=ax1z.transAxes, bbox=bbox)
    # Save as png, because pdf is insanely large
    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig('stress3D_{}_{}.png'.format(wf,nff), dpi=150)
    t_C = time.time()
    print("Time to calculate pressure: {:.3f} s".format(t_A-t_0))
    print("Time to plot pressures:     {:.3f} s".format(t_B-t_A))
    print("Time to save plots:         {:.3f} s".format(t_C-t_B))
    return




def plot_mass_slices():
    ''' Plot four slices of the mass density.
    These include two polarizations (mj=0 and mj=1)
    and two zero axes (x and z).
    '''
    # Set things up to make the four panels
    D = Density()
    nrows,ncols=1,4
    fig = py.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = py.subplot(nrows,ncols,1, aspect='equal')
    ax2 = py.subplot(nrows,ncols,2, aspect='equal')
    ax3 = py.subplot(nrows,ncols,3, aspect='equal')
    ax4 = py.subplot(nrows,ncols,4, aspect='equal')
    # Plot the masses
    _ = plot_one_mass_slice(D, ax1, mj=1, zero_axis='z', ylabel=True)
    _ = plot_one_mass_slice(D, ax2, mj=0, zero_axis='z', ylabel=False)
    _ = plot_one_mass_slice(D, ax3, mj=1, zero_axis='x', ylabel=True)
    c = plot_one_mass_slice(D, ax4, mj=0, zero_axis='x', ylabel=False)
    # Save
    fig.patch.set_alpha(0)
    fig.savefig('mass.pdf')
    return

def plot_one_mass_slice(D, ax, zero_axis='z', mj=0, ylabel=False):
    ''' Plot one of the four mass slices. '''
    # Set up the arrays for the plotted axes
    b = np.linspace(-2.12, 2.12, 400)
    if(zero_axis=='z'):
        x,y,z = b,b,0
        xlabel = r'$x$ (fm)'
        ylabel = r'$y$ (fm)'
    elif(zero_axis=='x'):
        x,y,z = 0,b,b
        xlabel = r'$y$ (fm)'
        ylabel = r'$z$ (fm)'
    elif(zero_axis=='y'):
        x,y,z = b,0,b
        xlabel = r'$x$ (fm)'
        ylabel = r'$z$ (fm)'
    else:
        raise ValueError("Invalid value for zero_azis: {}.".format(zero_axis))
    # Obtain the density. Squeeze everything down to 2-dimensional grids
    MU = np.squeeze( D.mass_3D_U(x,y,z) )
    MT = np.squeeze( D.mass_3D_T(x,y,z) )
    if(mj==0):
        M = MU + 2/3*MT
    elif(mj==1 or mj==-1):
        M = MU - 1/3*MT
    else:
        raise ValueError("mj={:d} is not a valid spin.".format(mj))
    # Plot the slice
    vmax = MU.max() + abs(MT).max()
    c = ax.pcolormesh(b, b, M.T, vmin=0, vmax=vmax, cmap='magma', shading='gouraud')
    # Label and leave
    ax.annotate(
            r'$m_j={:d},\, {}=0$'.format(mj,zero_axis),
            (0.025,0.025), xycoords='axes fraction',
            color='white'
            )
    ax.set_xlabel(xlabel)
    if(ylabel):
        ax.set_ylabel(ylabel)
    else:
        ax.get_yaxis().set_ticks([])
    return c

def plot_stress_slice(zero_axis='z', mj=0):
    ''' Plot a 2D slice of the 3D stresses, with one of the three coordinate
    axes set to zero.
    '''
    # Set up the arrays for the plotted axes
    b = np.linspace(-2.12, 2.12, 200)
    if(zero_axis=='z'):
        x,y,z = b,b,0
        xlabel = r'$x$ (fm)'
        ylabel = r'$y$ (fm)'
    elif(zero_axis=='x'):
        x,y,z = 0,b,b
        xlabel = r'$y$ (fm)'
        ylabel = r'$z$ (fm)'
    elif(zero_axis=='y'):
        x,y,z = b,0,b
        xlabel = r'$x$ (fm)'
        ylabel = r'$z$ (fm)'
    else:
        raise ValueError("Invalid value for zero_azis: {}.".format(zero_axis))
    # Obtain the densities. Squeeze everything down to 2-dimensional grids
    D = Density()
    TijU   = np.squeeze( D.stress_3D_U(x,y,z) )
    TijT   = np.squeeze( D.stress_3D_T(x,y,z) )
    rhat   = np.squeeze( make_rhat(x,y,z)     )
    phihat = np.squeeze( make_phihat(x,y,z)   )
    zhat   = np.squeeze( make_zhat(x,y,z)     )
    prU = np.einsum('xyij,xyi,xyj->xy', TijU, rhat, rhat)
    ptU = np.einsum('xyij,xyi,xyj->xy', TijU, phihat, phihat)
    pzU = np.einsum('xyij,xyi,xyj->xy', TijU, zhat, zhat)
    prT = np.einsum('xyij,xyi,xyj->xy', TijT, rhat, rhat)
    ptT = np.einsum('xyij,xyi,xyj->xy', TijT, phihat, phihat)
    pzT = np.einsum('xyij,xyi,xyj->xy', TijT, zhat, zhat)
    if(mj==0):
        pr = prU + 2/3*prT
        pt = ptU + 2/3*ptT
        pz = pzU + 2/3*pzT
    elif(mj==1 or mj==-1):
        pr = prU - 1/3*prT
        pt = ptU - 1/3*ptT
        pz = pzU - 1/3*pzT
    else:
        raise ValueError("mj={:d} is not a valid spin.".format(mj))
    vmax = max( abs(pr).max(), abs(pt).max(), abs(pz).max() )
    # Create the plot canvas
    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*8,nrows*8), layout='constrained')
    ax1 = py.subplot(nrows,ncols,1, aspect='equal')
    ax2 = py.subplot(nrows,ncols,2, aspect='equal')
    ax3 = py.subplot(nrows,ncols,3, aspect='equal')
    # Plot the stresses
    c1 = ax1.pcolormesh(b, b, pr.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    c2 = ax2.pcolormesh(b, b, pt.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    c3 = ax3.pcolormesh(b, b, pz.T, vmin=-vmax, vmax=vmax, cmap='vanimo', shading='gouraud')
    # Color bar
    cbar = fig.colorbar(c3)
    cbar.set_label(r'Stress (GeV/fm$^3$)')
    # Labels
    ax1.set_title('Radial stress')
    ax2.set_title('Azimuthal stress')
    ax3.set_title('$z$-direction stress')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel(xlabel)
        ax.annotate(
                r'$m_j={:d},\, {}=0$'.format(mj,zero_axis),
                (0.025,0.025), xycoords='axes fraction',
                color='white'
                )
    ax1.set_ylabel(ylabel)
    for ax in [ax2,ax3]:
        ax.get_yaxis().set_ticks([])
    # Save
    fig.patch.set_alpha(0)
    fig.savefig('stress_mj{:d}_{}0.pdf'.format(mj,zero_axis))
    return


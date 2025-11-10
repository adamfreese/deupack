import numpy as np
import deupack as dp

# Plotting things
import matplotlib as mpl
import matplotlib.pyplot as py
import matplotlib.ticker as mticker
mpl.rc('font',size=26,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")

def select_mff(name, dl2):
    if(name=='AU'):
        F = dp.mff.AU(np.sqrt(dl2))
    elif(name=='AT'):
        F = dp.mff.AT(np.sqrt(dl2))
    elif(name=='J'):
        F = dp.mff.J(np.sqrt(dl2))
    elif(name=='DU'):
        #F = dp.mff.DU(np.sqrt(dl2))#, DN=dp.mff.nucleon.DN_fc)
        F = dp.mff.DU(np.sqrt(dl2), DN=dp.mff.nucleon.DN_hz)
    elif(name=='DT1'):
        #F = dp.mff.DT1(np.sqrt(dl2))#, DN=dp.mff.nucleon.DN_fc)
        F = dp.mff.DT1(np.sqrt(dl2), DN=dp.mff.nucleon.DN_hz)
    elif(name=='DT2'):
        F = dp.mff.DT2(np.sqrt(dl2))
    else:
        F = dl2 * 0
    return F

namelabel = {
        'AU'  : r'$A_U(\varDelta^2)$',
        'AT'  : r'$A_T(\varDelta^2)$',
        'J'   : r'$J(\varDelta^2)$',
        'DU'  : r'$D_U(\varDelta^2)$',
        'DT1' : r'$D_{T1}(\varDelta^2)$',
        'DT2' : r'$D_{T2}(\varDelta^2)$'
        }

def plot_one_mff(ax, name):
    # MFF from other papers
    df_wc = dp.mff.wim.make_wimffs()
    fc_hz = dp.mff.hz.make_hzmffs()
    df_jp = dp.mff.pegg.make_peggmffs()
    F_wc = df_wc[name]
    F_hz = fc_hz[name]
    F_jp = df_jp[name]
    dl2_wc = df_wc['Delta2']
    dl2_hz = fc_hz['Delta2']
    dl2_jp = df_jp['Delta2']
    # Our MFF
    dl2 = np.geomspace(1e-6, 1e1, 666)
    F = select_mff(name, dl2)
    # Plot
    ax.plot(dl2,    F,    '-',  linewidth=2, color='xkcd:hunter green', label=r'Ours')
    ax.plot(dl2_wc, F_wc, '--', linewidth=2, color='xkcd:dark lavender',label=r'Freese and Cosyn')
    ax.plot(dl2_hz, F_hz, '-.', linewidth=2, color='xkcd:cobalt', label=r'He and Zahed')
    ax.plot(dl2_jp, F_jp, ':',  linewidth=2, color='xkcd:coral',  label=r'Panteleva \textsl{et al.}')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(namelabel[name])
    ax.set_xscale('log')
    if(name=='DT1'):
        ax.set_ylim((-560,56))
    ax.set_xlim((1e-6,10))
    return

def plot_comparisons():
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax_AU  = py.subplot(nrows,ncols,1)
    ax_AT  = py.subplot(nrows,ncols,2)
    ax_J   = py.subplot(nrows,ncols,3)
    ax_DU  = py.subplot(nrows,ncols,4)
    ax_DT1 = py.subplot(nrows,ncols,5)
    ax_DT2 = py.subplot(nrows,ncols,6)
    plot_one_mff(ax_AU,  'AU')
    plot_one_mff(ax_AT,  'AT')
    plot_one_mff(ax_J,   'J')
    plot_one_mff(ax_DU,  'DU')
    plot_one_mff(ax_DT1, 'DT1')
    plot_one_mff(ax_DT2, 'DT2')
    l = ax_AU.legend(prop = { 'size' : 24 }, loc=3)
    fig.patch.set_alpha(0)
    fig.savefig('derp.pdf')
    fig.savefig('deuteron_mff_comparison.pdf')
    return

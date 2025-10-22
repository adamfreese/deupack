import numpy as np
import deupack as dp

# Plotting things
import matplotlib as mpl
import matplotlib.pyplot as py
import matplotlib.ticker as mticker
mpl.rc('font',size=26,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")

def plot_one_mff(name, logscale=False):
    wimdf = dp.mff.wim.make_wimffs()
    hzdf = dp.mff.hezahed.make_hzmffs()
    Delta2 = wimdf['Delta2']
    if(name=='AU'):
        F_our = dp.mff.AU(np.sqrt(Delta2))
    elif(name=='AT'):
        F_our = dp.mff.AT(np.sqrt(Delta2))
    elif(name=='J'):
        F_our = dp.mff.J(np.sqrt(Delta2))
    elif(name=='DU'):
        #F_our = dp.mff.DU(np.sqrt(Delta2))#, DN=dp.mff.nucleon.DN_fc)
        F_our = dp.mff.DU(np.sqrt(Delta2), DN=dp.mff.nucleon.DN_hz)
    elif(name=='DT1'):
        #F_our = dp.mff.DT1(np.sqrt(Delta2))#, DN=dp.mff.nucleon.DN_fc)
        F_our = dp.mff.DT1(np.sqrt(Delta2), DN=dp.mff.nucleon.DN_hz)
    elif(name=='DT2'):
        F_our = dp.mff.DT2(np.sqrt(Delta2))
    F_wim = wimdf[name]
    F_hez = hzdf[name]
    dl2_hz = hzdf['Delta2']
    namelabel = {
            'AU'  : r'$A_U(\varDelta^2)$',
            'AT'  : r'$A_T(\varDelta^2)$',
            'J'   : r'$J(\varDelta^2)$',
            'DU'  : r'$D_U(\varDelta^2)$',
            'DT1' : r'$D_{T1}(\varDelta^2)$',
            'DT2' : r'$D_{T2}(\varDelta^2)$'
            }
    # Make the plot
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
    if(logscale):
        F_our = abs(F_our)
        F_wim = abs(F_wim)
    ax.plot(Delta2, F_our, '-',  color='xkcd:forest green',  label=r'Ours')
    ax.plot(Delta2, F_wim, '--', color='xkcd:rich purple',   label=r'Freese and Cosyn')
    ax.plot(dl2_hz, F_hez, '-.', color='xkcd:midnight blue', label=r'He and Zahed')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(namelabel[name])
    _ = ax.legend(prop = { 'size' : 20 })
    ax.set_xscale('log')
    if(logscale):
        ax.set_yscale('log')
    fig.patch.set_alpha(0)
    fig.savefig('derp.pdf')
    fig.savefig('{}.pdf'.format(name))
    return

def plot_AU():
    wimdf = dp.mff.wim.make_wimffs()
    Delta2 = wimdf['Delta2']
    AU = dp.mff.AU(np.sqrt(Delta2))
    AU_wim = wimdf['AU']
    # Make the plot
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
    ax.plot(Delta2, AU,     '-',  color='xkcd:forest green', label=r'Ours')
    ax.plot(Delta2, AU_wim, '--', color='xkcd:rich purple',  label=r'Freese and Cosyn')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$A_U(\varDelta^2)$')
    _ = ax.legend(prop = { 'size' : 20 })
    ax.set_xscale('log')
    fig.patch.set_alpha(0)
    fig.savefig('derp.pdf')
    fig.savefig('AU.pdf')
    return

def plot_AT():
    wimdf = dp.mff.wim.make_wimffs()
    Delta2 = wimdf['Delta2']
    AU = dp.mff.AT(np.sqrt(Delta2))
    AU_wim = wimdf['AT']
    # Make the plot
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*6), layout='constrained')
    ax = py.subplot(nrows,ncols,1)
    ax.plot(Delta2, AU,     '-',  color='xkcd:forest green', label=r'Ours')
    ax.plot(Delta2, AU_wim, '--', color='xkcd:rich purple',  label=r'Freese and Cosyn')
    ax.set_xlabel(r'$\varDelta^2$ (GeV$^2$)')
    ax.set_ylabel(r'$A_T(\varDelta^2)$')
    _ = ax.legend(prop = { 'size' : 20 })
    ax.set_xscale('log')
    fig.patch.set_alpha(0)
    fig.savefig('derp.pdf')
    fig.savefig('AT.pdf')
    return

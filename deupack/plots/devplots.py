# paperplots.py
#
# Created 2026.03.12 by Adam Freese, for some personal purposes.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmasher as cmr

from .. import emtff
from ..density import NucleonDensity as Density
from .density3d import multidensity3d

mpl.rc('font',size=30,family='cmr10',weight='normal')
mpl.rc('text',usetex=True)
mpl.rc('text.latex', preamble=r"\usepackage{bm,amsmath,amssymb,amsfonts,mathrsfs}")
plt.rcParams["axes.formatter.use_mathtext"] = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pressure():
    # Fixed parameters for the visualization
    nff='ba'; nb=151; bmax=0.84
    # Get the pressures
    D = Density(nff=nff, nb=nb, bmax=bmax)
    pr = D.radial_pressure(pol=0)
    pt = D.polar_pressure( pol=0)
    # Make some labels
    labels = [
            r'radial pressure',
            r'tangential pressure'
            ]
    # Prepare figure
    nrows,ncols=1,2
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    # Use the custom multi-3D plotter
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr, pt,
                   labels=labels,
                   clabel=r'Pressure (GeV/fm$^3$)',
                   decay=1, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('nucleon_pressure.pdf')
    return

def pressure_separated():
    # Fixed parameters for the visualization
    nb=151; bmax=0.84
    # Get the pressures
    D = Density(nff='mit', nb=nb, bmax=bmax)
    Dq = Density(nff='mitq', nb=nb, bmax=bmax)
    Dg = Density(nff='mitg', nb=nb, bmax=bmax)
    pr = D.radial_pressure(pol=0)
    pt = D.polar_pressure( pol=0)
    prq = Dq.radial_pressure(pol=0)
    ptq = Dq.polar_pressure( pol=0)
    prg = Dg.radial_pressure(pol=0)
    ptg = Dg.polar_pressure( pol=0)
    # Make some labels
    labels = [
            r'total radial pressure',
            r'quark radial pressure',
            r'gluon radial pressure',
            r'total tangential pressure',
            r'quark tangential pressure',
            r'gluon tangential pressure'
            ]
    # Prepare figure
    nrows,ncols=2,3
    fig = plt.figure(figsize=(ncols*10,nrows*10+1))
    # Use the custom multi-3D plotter
    multidensity3d(fig, D.x, D.x, D.x,
                   nrows, ncols,
                   pr, prq, prg, pt, ptq, ptg,
                   labels=labels,
                   clabel=r'Pressure (GeV/fm$^3$)',
                   decay=1, opacity=0.69, cmap=cmr.fusion_r,
                   projections=True, divergent=True, s=1)
    fig.savefig('nucleon_pressure_separated.pdf')
    return

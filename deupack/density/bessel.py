# bessel.py
# Created 2026.09.23 by Adam Freese
# moved several functions from density3d.py to this file
#
# Integrands for Bessel transforms

import numpy as np
import pandas as pd

from scipy.special import spherical_jn as jn

from ..constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Integrand functions
# Need to make these separate functions to use quad_vec with workers

def massU_integrand(k, b, AU, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN
    bessel = jn(0, k*b/hbar)
    form = AU(k)
    return common * unique * bessel * form

def massT_integrand(k, b, AT, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = k**2/(2*mN)
    bessel = jn(2, k*b/hbar)
    form = AT(k)
    return common * unique * bessel * form

def momentum_integrand(k, b, J, S):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = k/2
    bessel = jn(1, k*b/hbar)
    form = J(k) - S(k)
    return common * unique * bessel * form

def flux_integrand(k, b, J, S):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = k/2
    bessel = jn(1, k*b/hbar)
    form = J(k) + S(k)
    return common * unique * bessel * form

def pressureU_integrand(k, b, DU, cU, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DU(k) + 2*mN*cU(k)
    return common * unique * bessel * form

def pressureT1_integrand(k, b, DT1, cT1, mN):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def pressureT2_integrand(k, b, DT2, cT2, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT2(k) + 2*mN*cT2(k)
    return common * unique * bessel * form

def shearU_integrand(k, b, DU, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DU(k)
    return common * unique * bessel * form

def shearT1_integrand(k, b, DT1, mN):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

def shearT2_integrand(k, b, DT2, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT2(k)
    return common * unique * bessel * form

def shearA_integrand(k, b, sbar, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*sbar(k)
    return common * unique * bessel * form

# T1 stress integrals for direct use (no differentiation) ~~~~~~~~~~~~~~~~~~~~~~

def pressureT1_integrand_direct(k, b, DT1, cT1, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -k**2/(8*mN**2)
    bessel = jn(2, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def shearT1_integrand_direct(k, b, DT1, mN, norder):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = (-1)**(norder//2) * k**2/(8*mN**2)
    bessel = jn(norder, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

# Integrands for force distributions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def f0_integrand(k, b, cU, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(1, k*b/hbar)
    form = cU(k)
    return common * unique * bessel * form

def f2_integrand(k, b, cT1, cT2, sbar, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(1, k*b/hbar)
    form = cT2(k) - k**2/(8*mN**2)*sbar(k) - k**2/(20*mN**2)*(cT1(k)-sbar(k))
    return common * unique * bessel * form

def f3_integrand(k, b, cT1, sbar, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(3, k*b/hbar)
    form = k**2/(8*mN**2)*(cT1(k)-sbar(k))
    return common * unique * bessel * form

def f2_integrandSym(k, b, cT1, cT2, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(1, k*b/hbar)
    form = cT2(k)  - k**2/(20*mN**2)*(cT1(k))
    return common * unique * bessel * form

def f3_integrandSym(k, b, cT1, mN):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(3, k*b/hbar)
    form = k**2/(8*mN**2)*(cT1(k))
    return common * unique * bessel * form

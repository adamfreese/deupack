# moments.py
# Created 2026.02.03 by Adam Freese
#
# Contains code to find radii and quadrupole moments of the deuteron EMT-FFs

import numpy as np

from scipy.integrate import quad

from ..constants import mN, Md, hbar

# Choosers
from ..wf.chooser import choose_wf
from .nucleon.chooser import choose_nff

from .deuteron import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Radii

def mass_radius_squared(nff='ba', wf='av18'):
    ''' Mean squared mass radius is the sum of wave function and nucleon
    contributions. Result is given in fm**2.
    '''
    _dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    r2 = _dwf.radius_squared() + _nff.mass_radius_squared()
    return r2

def mechanical_radius_squared(nff='ba', wf='av18'):
    ''' Mechanical radius, defined as the mean squared radius of the radial
    pressure distribution. This can be obtained by dividing the b2 moment
    by the integrated radial pressure.
    '''
    r2 = mechanical_b2_moment(nff=nff, wf=wf) / mechanical_charge(nff=nff, wf=wf)
    return r2*hbar**2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quadrupole moments

def mass_quadrupole_momenmt(nff='ba', wf='av18'):
    ''' Mass quadrupole moment, in fm**2. '''
    # There are actually two equivalent ways of getting this
    # Method 1: from the wave function
    _wf = choose_wf(wf)
    Qd1 = _wf.quadrupole()
    # Method 2: from the EMT-FF AT
    # (The result doesn't actually depend on the NFF)
    Qd2 = AT(0, nff=nff, wf=wf) / (2*mN)**2 * hbar**2
    # Return both just to show they're equal
    return Qd1, Qd2

def mechanical_quadrupole_moment(nff='ba', wf='av18'):
    # TODO
    return 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other moments

def mechanical_charge(nff='ba', wf='av18'):
    ''' Thinking of the radial pressure as a sort of density, this integrates
    said density to get a sort of charge. It's effectively just a normalization
    for defining the mechanical radius and quadrupole moments, and does not
    correspond to a physically meaningful charge.
    '''
    # Part of the answer depends on the integral of DU(k).
    # np.trapz is good enough for our purposes, if a lot of points are used.
    k_array = np.geomspace(1e-6, 100, 666)
    DU_array = DU(k_array, nff=nff, wf=wf)
    DU_integral = np.trapz(DU_array, x=k_array**2)
    cU0 = cU(0, nff=nff, wf=wf)
    return -DU_integral/(4*Md) - Md*cU0

def mechanical_b2_moment(nff='ba', wf='av18'):
    ''' The integral of b**2 times the radial pressure distribution. '''
    DU_term = -3/(2*Md) * DU(0, nff=nff, wf=wf)
    cU_derivative_term = - Md * _6_dcUdt(nff=nff, wf=wf)
    return DU_term + cU_derivative_term

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper methods

def _6_dcUdt(nff='ba', wf='av18'):
    ''' Derivative, in GeV**-2. '''
    _wf = choose_wf(wf)
    _nff = choose_nff(nff)
    def integrand1(r):
        return _wf.u(r)*_wf.u2(r) + _wf.w(r)*_wf.w2(r) - 6*_wf.w(r)**2/r**2
    def integrand2(r):
        return r**2*(
                _wf.u(r)*_wf.u2(r) + _wf.w(r)*_wf.w2(r)
                - _wf.u1(r)**2 - _wf.w1(r)**2
                )
    coef1 = quad(integrand1, 0, np.inf)[0] / (6*mN**2)
    coef2 = 3/20*(1/4*quad(integrand2, 0, np.inf)[0] - _wf.Pd()) / mN**2
    r2 = coef1 * _nff.mass_radius_squared() + coef2
    # TODO: cbarN contributions (zero right now, but might matter later)
    return r2 * 2 # factor 2 from sum over nucleons

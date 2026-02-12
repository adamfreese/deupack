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
    by the integrated radial pressure. Result is given in fm**2.
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
    ''' Mechanical quadrupole moment, defined as the mean value of 3*z**2-b**2
    over the radial pressure distribution. This can be obtained by dividng the
    3*z**2-b**2 moment by the integrated radial presure.
    Result is given in fm**2.
    '''
    Qmech = (
            mechanical_3z2_minus_b2_moment(nff=nff, wf=wf)
            / mechanical_charge(nff=nff, wf=wf)
            )
    return Qmech*hbar**2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other moments

def mechanical_charge(nff='ba', wf='av18'):
    ''' Thinking of the radial pressure as a sort of density, this integrates
    said density to get a sort of charge. It's effectively just a normalization
    for defining the mechanical radius and quadrupole moments, and does not
    correspond to a physically meaningful charge.
    '''
    # Part of the answer depends on the integral of DU(k).
    # np.trapezoid is good enough for our purposes, if a lot of points are used.
    k_array = np.geomspace(1e-6, 100, 666)
    DU_array = DU(k_array, nff=nff, wf=wf)
    DU_integral = np.trapezoid(DU_array, x=k_array**2)
    cU0 = cU(0, nff=nff, wf=wf)
    return -DU_integral/(4*Md) - Md*cU0

def mechanical_b2_moment(nff='ba', wf='av18'):
    ''' The integral of b**2 times the radial pressure distribution.
    Numerator of the mechanical radius.
    '''
    DU_term = -3/(2*Md) * DU(0, nff=nff, wf=wf)
    cU_derivative_term = 6 * Md * _emtff_derivative(cU, nff=nff, wf=wf)
    return DU_term + cU_derivative_term

def mechanical_3z2_minus_b2_moment(nff='ba', wf='av18'):
    ''' The integral of 3*z**2-b**2 times the radial pressure distribution.
    Numerator of the mechanical quadrupole moment.
    '''
    # T1 terms ~~~~~~~~~~~~~~~~~~~~~~~~~
    # Part of the answer depends on the integral of DT1(k).
    # np.trapezoid is good enough for our purposes, if a lot of points are used.
    k_array = np.geomspace(1e-6, 100, 666)
    DT1_array = DT1(k_array, nff=nff, wf=wf)
    DT1_integral = np.trapezoid(DT1_array, x=k_array**2)
    cT10 = cT1(0, nff=nff, wf=wf)
    T1_terms = (-3*DT1_integral/(10*Md) - Md*cT10) / (Md**2)
    # T2 terms ~~~~~~~~~~~~~~~~~~~~~~~~~
    DT2_term = -3/(2*Md) * DT2(0, nff=nff, wf=wf)
    cT2_derivative_term = 6 * Md * _emtff_derivative(cT2, nff=nff, wf=wf)
    T2_terms = (DT2_term + cT2_derivative_term) * 2/(15*Md)
    return T1_terms + T2_terms

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper methods

def _emtff_derivative(F, nff='ba', wf='av18'):
    h = 1e-4
    k2 = np.array([0, h])
    k = np.sqrt(k2)
    FF = F(k, nff=nff, wf=wf)
    dFdk2 = (FF[1] - FF[0])/h
    return dFdk2

# density.py
# Created 2025.11.18 by Adam Freese
#
# This file computes mechanical densities for the deuteron.

import numpy as np

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline

from .constants import mN, hbar

from .mff import deuteron as mff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density class

class Density:
    ''' A class for the calculation of deuteorn densities.
    This is implemented as a class so that lookup tables for mechanical form
    factors can be cached, and so that the user can create different objects
    with different MFFs in their cache.
    '''

    def __init__(self,
                 nff='ba',
                 wf='av18',
                 nk=100,
                 kmin=1e-6, # GeV
                 kmax=10,   # GeV
                 ):
        self.nff  = nff
        self.wf   = wf
        self.nk   = nk
        self.kmin = kmin
        self.kmax = kmax
        self._init_mff_table()
        return

    # 1D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mass_1D_U(self, b):
        ''' Unpolarized part of mass density.
        b should be numpy.array in fm.
        '''
        integral = quad_vec(_massU_integrand, self.kmin, self.kmax,
                            args=(b, self.AU),
                            workers=8)[0]
        return integral

    def mass_1D_T(self, b):
        ''' Tensor-polarized part of mass density.
        b should be numpy.array in fm.
        '''
        integral = quad_vec(_massT_integrand, self.kmin, self.kmax,
                            args=(b, self.AT),
                            workers=8)[0]
        return integral

    def momentum_1D(self, b):
        ''' Momentum density (sans sxb factor).
        b should be numpy.array in fm.
        '''
        integral = quad_vec(_momentum_integrand, self.kmin, self.kmax,
                            args=(b, self.J, self.S),
                            workers=8)[0]
        return integral

    def flux_1D(self, b):
        ''' Mass flux density (sans sxb factor).
        b should be numpy.array in fm.
        '''
        integral = quad_vec(_flux_integrand, self.kmin, self.kmax,
                            args=(b, self.J, self.S),
                            workers=8)[0]
        return integral

    def piso_1D_U(self, b):
        # TODO: docstring
        integral = quad_vec(_pisoU_integrand, self.kmin, self.kmax,
                            args=(b, self.DU, self.cU),
                            workers=8)[0]
        return integral

    def piso_1D_T1(self, b):
        # TODO: docstring
        # NOTE: this is the Polyakov-Sun density, which must be differentiated
        integral = quad_vec(_pisoT1_integrand, self.kmin, self.kmax,
                            args=(b, self.DT1, self.cT1),
                            workers=8)[0]
        return integral

    def piso_1D_T2(self, b):
        # TODO: docstring
        integral = quad_vec(_pisoT2_integrand, self.kmin, self.kmax,
                            args=(b, self.DT2, self.cT2),
                            workers=8)[0]
        return integral

    def pani_1D_U(self, b):
        # TODO: docstring
        integral = quad_vec(_paniU_integrand, self.kmin, self.kmax,
                            args=(b, self.DU),
                            workers=8)[0]
        return integral

    def pani_1D_T1(self, b):
        # TODO: docstring
        # NOTE: this is the Polyakov-Sun density, which must be differentiated
        integral = quad_vec(_paniT1_integrand, self.kmin, self.kmax,
                            args=(b, self.DT1),
                            workers=8)[0]
        return integral

    def pani_1D_T2(self, b):
        # TODO: docstring
        integral = quad_vec(_paniT2_integrand, self.kmin, self.kmax,
                            args=(b, self.DT2),
                            workers=8)[0]
        return integral

    # 3D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO

    # Internal metthods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _init_mff_table(self):
        k   = np.geomspace(self.kmin, self.kmax, self.nk)
        AU  = mff.AU( k, wf=self.wf, nff=self.nff)
        AT  = mff.AT( k, wf=self.wf, nff=self.nff)
        DU  = mff.DU( k, wf=self.wf, nff=self.nff)
        DT1 = mff.DT1(k, wf=self.wf, nff=self.nff)
        DT2 = mff.DT2(k, wf=self.wf, nff=self.nff)
        cU  = mff.cU( k, wf=self.wf, nff=self.nff)
        cT1 = mff.cT1(k, wf=self.wf, nff=self.nff)
        cT2 = mff.cT2(k, wf=self.wf, nff=self.nff)
        J   = mff.J(  k, wf=self.wf, nff=self.nff)
        S   = mff.S(  k, wf=self.wf, nff=self.nff)
        self.AU  = CubicSpline(k, AU)
        self.AT  = CubicSpline(k, AT)
        self.DU  = CubicSpline(k, DU)
        self.DT1 = CubicSpline(k, DT1)
        self.DT2 = CubicSpline(k, DT2)
        self.cU  = CubicSpline(k, cU)
        self.cT1 = CubicSpline(k, cT1)
        self.cT2 = CubicSpline(k, cT2)
        self.J   = CubicSpline(k, J)
        self.S   = CubicSpline(k, S)
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Integrand functions
# Need to make these separate functions to use quad_vec with workers

def _massU_integrand(k, b, AU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN
    bessel = jn(0, k*b/hbar)
    form = AU(k)
    return common * unique * bessel * form

def _massT_integrand(k, b, AT):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -k**2/(4*mN)
    bessel = jn(2, k*b/hbar)
    form = AT(k)
    return common * unique * bessel * form

def _momentum_integrand(k, b, J, S):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = k/2
    bessel = jn(1, k*b/hbar)
    form = J(k) - S(k)
    return common * unique * bessel * form

def _flux_integrand(k, b, J, S):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = k/2
    bessel = jn(1, k*b/hbar)
    form = J(k) + S(k)
    return common * unique * bessel * form

def _pisoU_integrand(k, b, DU, cU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DU(k) + 2*mN*cU(k)
    return common * unique * bessel * form

def _pisoT1_integrand(k, b, DT1, cT1):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def _pisoT2_integrand(k, b, DT2, cT2):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT2(k) + 2*mN*cT2(k)
    return common * unique * bessel * form

def _paniU_integrand(k, b, DU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DU(k)
    return common * unique * bessel * form

def _paniT1_integrand(k, b, DT1):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

def _paniT2_integrand(k, b, DT2):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT2(k)
    return common * unique * bessel * form

# T1 stress integrals for direct use (no differentiation)

def _pisoT1_integrand_direct(k, b, DT1, cT1):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -k**2/(8*mN**2)
    bessel = jn(2, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def _paniT1_integrand_direct(k, b, DT1, norder):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = (-1)**(norder//2) * k**2/(8*mN**2)
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

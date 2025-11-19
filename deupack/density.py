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

    def mass_3D_U(self, x, y, z):
        ''' Three-dimensional mass density of the deuteron.
        Unpolarized contribution.
        '''
        x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
        b = np.sqrt(x_**2 + y_**2 + z_**2)
        return self.mass_1D_U(b)

    def mass_3D_T(self, x, y, z):
        ''' Three-dimensional mass density of the deuteron.
        Tensor-polarized contribution.
        '''
        x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
        b = np.sqrt(x_**2 + y_**2 + z_**2)
        scalar = self.mass_1D_T(b)
        e = make_zhat(x,y,z)
        Y2 = make_Y2(x,y,z)
        harmonics = np.einsum('zxyi,zxyj,zxyij->zxy', e, e, Y2)
        return harmonics * scalar



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
# Functions to make harmonic tensors

def make_rhat(x, y, z):
    # TODO: docstring
    eps = 1e-9 # to regulate division by zero
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    r_ = np.sqrt(x_**2 + y_**2 + z_**2 + eps)
    rhat = np.zeros(x_.shape+(3,))
    rhat[...,0] = z_/r_
    rhat[...,1] = x_/r_
    rhat[...,2] = y_/r_
    return rhat

def make_zhat(x, y, z):
    # TODO: docstring
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    zhat = np.zeros(x_.shape+(3,))
    zhat[...,0] = 1
    return zhat

def make_kronecker(x, y, z):
    # TODO: docstring
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    kronecker = np.zeros(x_.shape+(3,3))
    kronecker[...,0,0] = 1
    kronecker[...,1,1] = 1
    kronecker[...,2,2] = 1
    return kronecker

def make_Y0(x, y, z):
    # TODO: docstring
    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    Y0 = np.ones(x_.shape)
    return Y0

def make_Y1(x, y, z):
    # TODO: docstring
    return make_rhat(x, y, z)

def make_Y2(x, y, z):
    # TODO: docstring
    r = make_rhat(x,y,z)
    d = make_kronecker(x,y,z)
    rr = np.einsum('zxyi,zxyj->zxyij', r, r)
    return rr - d/3

def make_Y3(x, y, z):
    # TODO: docstring
    r = make_rhat(x,y,z)
    d = make_kronecker(x,y,z)
    rrr = np.einsum('zxyi,zxyj,zxyk->zxyijk', r, r, r)
    dr1 = np.einsum('zxyij,zxyk->zxyijk', d, r)
    dr2 = np.einsum('zxyki,zxyj->zxyijk', d, r)
    dr3 = np.einsum('zxyjk,zxyi->zxyijk', d, r)
    # Test to make sure these are distinct permutations ... passed!
    #print( (dr1 - dr2).max(), (dr1 - dr2).min())
    #print( (dr2 - dr3).max(), (dr2 - dr3).min())
    #print( (dr3 - dr1).max(), (dr3 - dr1).min())
    return rrr - (dr1+dr2+dr3)/3

def make_Y4(x, y, z):
    # TODO: docstring
    r = make_rhat(x,y,z)
    d = make_kronecker(x,y,z)
    rrrr = np.einsum('zxyi,zxyj,zxyk,zxyl->zxyijkl', r, r, r, r)
    drr1 = np.einsum('zxyij,zxyk,zxyl->zxyijkl', d, r, r)
    drr2 = np.einsum('zxykl,zxyi,zxyj->zxyijkl', d, r, r)
    drr3 = np.einsum('zxyik,zxyj,zxyl->zxyijkl', d, r, r)
    drr4 = np.einsum('zxyjk,zxyi,zxyl->zxyijkl', d, r, r)
    drr5 = np.einsum('zxyil,zxyj,zxyk->zxyijkl', d, r, r)
    drr6 = np.einsum('zxyjl,zxyi,zxyk->zxyijkl', d, r, r)
    # Test to make sure these are distinct permutations ... passed!
    #print( (drr1 - drr2).max(), (drr1 - drr2).min())
    #print( (drr1 - drr3).max(), (drr1 - drr3).min())
    #print( (drr1 - drr4).max(), (drr1 - drr4).min())
    #print( (drr1 - drr5).max(), (drr1 - drr5).min())
    #print( (drr1 - drr6).max(), (drr1 - drr6).min())
    #print( (drr2 - drr3).max(), (drr2 - drr3).min())
    #print( (drr2 - drr4).max(), (drr2 - drr4).min())
    #print( (drr2 - drr5).max(), (drr2 - drr5).min())
    #print( (drr2 - drr6).max(), (drr2 - drr6).min())
    #print( (drr3 - drr4).max(), (drr3 - drr4).min())
    #print( (drr3 - drr5).max(), (drr3 - drr5).min())
    #print( (drr3 - drr6).max(), (drr3 - drr6).min())
    #print( (drr4 - drr5).max(), (drr4 - drr5).min())
    #print( (drr4 - drr6).max(), (drr4 - drr6).min())
    #print( (drr5 - drr6).max(), (drr5 - drr6).min())
    dd1 = np.einsum('zxyij,zxykl->zxyijkl', d, d)
    dd2 = np.einsum('zxyik,zxyjl->zxyijkl', d, d)
    dd3 = np.einsum('zxyil,zxykj->zxyijkl', d, d)
    # Test to make sure these are distinct permutations ... passed!
    #print( (dd1 - dd2).max(), (dd1 - dd2).min())
    #print( (dd2 - dd3).max(), (dd2 - dd3).min())
    #print( (dd3 - dd1).max(), (dd3 - dd1).min())
    return (
            rrrr
            - (drr1 + drr2 + drr3 + drr4 + drr5 + drr6)/7
            + (dd1 + dd2 + dd3)/35
            )

# Peculiar tensors appearing in tensor-polarized stresses

def make_X2(x, y, z):
    # TODO: docstring
    Y2 = make_Y2(x, y, z)
    dl = make_kronecker(x, y, z)
    ijab = np.einsum('zxyij,zxyab->zxyijab', Y2, dl)
    abij = np.einsum('zxyab,zxyij->zxyijab', Y2, dl)
    aibj = np.einsum('zxyai,zxybj->zxyijab', Y2, dl)
    ajbi = np.einsum('zxyaj,zxybi->zxyijab', Y2, dl)
    bjai = np.einsum('zxybj,zxyai->zxyijab', Y2, dl)
    biaj = np.einsum('zxybi,zxyaj->zxyijab', Y2, dl)
    term1 = -(ijab + abij + aibj + ajbi + bjai + abiaj)/7
    term2 = -(ijab + abij)/3
    return term1 + term2

def make_X0(x, y, z):
    dl = make_kronecker(x, y, z)
    ijab = np.einsum('zxyij,zxyab->zxyijab', dl, dl)
    aibj = np.einsum('zxyai,zxybj->zxyijab', dl, dl)
    ajbi = np.einsum('zxyaj,zxybi->zxyijab', dl, dl)
    term1 = -(ijab + aibj + ajbi)/15
    term2 = -ijab/9
    return term1 + term2

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

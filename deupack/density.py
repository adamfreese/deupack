# density.py
# Created 2025.11.18 by Adam Freese
#
# This file computes mechanical densities for the deuteron.

import numpy as np
import pandas as pd

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline

from pathlib import Path

from .constants import mN, hbar
from . import mff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density class

# TODO: public-facing methods for momentum and mass flux densities

class Density:
    ''' A class for the calculation of deuteron densities.

    This is implemented as a class so that lookup tables for mechanical form
    factors can be cached, and so that the user can create different objects
    with different MFFs in their cache. Additionally, the density functions can
    themselves be cached on a per-object basis to reduce computation time when
    creating 3D density plots.
    '''

    def __init__(self,
                 wf='av18',
                 nff='ba',
                 nk=100,
                 nb=120,
                 bmax=2,    # fm
                 kmin=1e-6, # GeV
                 kmax=10,   # GeV
                 use_cache=True
                 ):
        self.wf   = wf
        self.nff  = nff
        self.nk   = nk
        self.nb   = nb
        self.bmax = bmax
        self.kmin = kmin
        self.kmax = kmax
        # Internal initializations of spatial variables and Bessel caches
        self._initialize_space()
        self._initialize_bessel()
        # attempt to find a cached lookup table on disk
        path = self._cache_path()
        if(path.is_file() and use_cache):
            self._load_mff_table(path)
        else:
            # if not found, make one
            self._init_mff_table(save_table=True)
        return

    # 3D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mass_density(self, pol='U'):
        ''' Mass density in GeV/fm**3. '''
        if(pol=='U'):
            return self._mass_bessel_U()
        elif(pol=='T'):
            b_dep = self._mass_bessel_T()
            theta_dep = 3/2*np.cos(self.theta)**2 - 1/2
            return b_dep*theta_dep
        elif(pol==0):
            return self.mass_density(pol='U') + 2/3*self.mass_density(pol='T')
        elif(pol==1 or pol==-1):
            return self.mass_density(pol='U') - 1/3*self.mass_density(pol='T')
        else:
            self._pol_error(pol)
        return

    def radial_pressure(self, pol='U'):
        ''' Radial pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() + 2/3*self._shear_bessel_U()
        elif(pol=='T'):
            b_dep = (
                    self._pressure_bessel_T1()
                    +
                    2/15*self._shear_bessel_T1(0)
                    +
                    4/21*self._shear_bessel_T1(2)
                    +
                    12/35*self._shear_bessel_T1(4)
                    +
                    self._pressure_bessel_T2()
                    +
                    2/3*self._shear_bessel_T2()
                    )
            theta_dep = 3/2*np.cos(self.theta)**2 - 1/2
            return b_dep*theta_dep
        elif(pol==0):
            return self.radial_pressure(pol='U') + 2/3*self.radial_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.radial_pressure(pol='U') - 1/3*self.radial_pressure(pol='T')
        else:
            self._pol_error(pol)
        return

    def lateral_pressure(self, pol='U'):
        ''' Lateral pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() - 1/3*self._shear_bessel_U()
        elif(pol=='T'):
            b_dep1 = (
                    self._pressure_bessel_T1()
                    -
                    2/15*self._shear_bessel_T1(0)
                    -
                    1/5*self._shear_bessel_T1(4)
                    -
                    self._pressure_bessel_T2()
                    -
                    2/3*self._shear_bessel_T2()
                    )
            b_dep0 = (
                    1/15*self._shear_bessel_T1(0)
                    -
                    2/21*self._shear_bessel_T1(2)
                    +
                    1/35*self._shear_bessel_T1(4)
                    +
                    1/2*self._pressure_bessel_T2()
                    -
                    2/3*self._shear_bessel_T2()
                    )
            theta_dep1 = 3/2*np.cos(self.theta)**2 - 1/2
            return theta_dep1*b_dep1 + b_dep0
        elif(pol==0):
            return self.lateral_pressure(pol='U') + 2/3*self.lateral_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.lateral_pressure(pol='U') - 1/3*self.lateral_pressure(pol='T')
        else:
            self._pol_error(pol)
        return

    def azimuthal_pressure(self, pol='U'):
        ''' Lateral pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() - 1/3*self._shear_bessel_U()
        elif(pol=='T'):
            b_dep1 = (
                    self._pressure_bessel_T1()
                    -
                    4/21*self._shear_bessel_T1(2)
                    -
                    1/7*self._shear_bessel_T1(4)
                    -
                    2*self._shear_bessel_T2()
                    )
            b_dep0 = (
                    -
                    1/15*self._shear_bessel_T1(0)
                    +
                    2/21*self._shear_bessel_T1(2)
                    -
                    1/35*self._shear_bessel_T1(4)
                    -
                    1/2*self._pressure_bessel_T2()
                    +
                    2/3*self._shear_bessel_T2()
                    )
            theta_dep1 = 3/2*np.cos(self.theta)**2 - 1/2
            return theta_dep1*b_dep1 + b_dep0
        elif(pol==0):
            return self.azimuthal_pressure(pol='U') + 2/3*self.azimuthal_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.azimuthal_pressure(pol='U') - 1/3*self.azimuthal_pressure(pol='T')
        else:
            self._pol_error(pol)
        return

    def symmetric_shear(self, pol='U'):
        ''' Symmetric shear in the r-theta directions, in GeV/fm**3. '''
        theta_dep = 0.5*np.sin(2*self.theta)
        b_dep = (
                -
                1/5*self._shear_bessel_T1(0)
                -
                1/7*self._shear_bessel_T1(2)
                +
                12/35*self._shear_bessel_T1(4)
                -
                3/2*self._pressure_bessel_T2()
                -
                self._shear_bessel_T2()
                )
        shear = theta_dep*b_dep
        if(pol=='U'):
            shear *= 0
        elif(pol=='T'):
            pass
        elif(pol==0):
            shear *= 2/3
        elif(pol==1 or pol==-1):
            shear *= -1/3
        else:
            self._pol_error(pol)
        return shear

    def pseudoradial_pressure(self, pol='U'):
        ''' Eigenpressure closest to the radial direction, in GeV/fm**3. '''
        pr = self.radial_pressure(pol=pol)
        pt = self.lateral_pressure(pol=pol)
        s = self.symmetric_shear(pol=pol)
        return 0.5*(pr+pt + np.sqrt((pr-pt)**2+4*s**2))

    def pseudolateral_pressure(self, pol='U'):
        ''' Eigenpressure closest to the lateral direction, in GeV/fm**3. '''
        pr = self.radial_pressure(pol=pol)
        pt = self.lateral_pressure(pol=pol)
        s = self.symmetric_shear()
        return 0.5*(pr+pt - np.sqrt((pr-pt)**2+4*s**2))

    # Bessel transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _mass_bessel_U(self):
        ''' Unpolarized part of mass density; b dependence.
        Uses internal spatial variables.
        '''
        if(self.bessel_aU is None):
            self.bessel_aU = quad_vec(_massU_integrand, self.kmin, self.kmax,
                                      args=(self.b, self.AU),
                                      )[0]
        return self.bessel_aU

    def _mass_bessel_T(self):
        ''' Tesnor-polarized part of mass density; b dependence.
        Uses internal spatial variables.
        '''
        if(self.bessel_aT is None):
            self.bessel_aT = quad_vec(_massT_integrand, self.kmin, self.kmax,
                                      args=(self.b, self.AT),
                                      )[0]
        return self.bessel_aT

    def _momentum_bessel(self, b):
        ''' Momentum density (sans sxb factor).
        Uses internal spatial variables.
        '''
        if(self.bessel_k is None):
            self.bessel_k = quad_vec(_momentum_integrand, self.kmin, self.kmax,
                                     args=(self.b, self.J, self.S),
                                     )[0]
        return self.bessel_k

    def _flux_bessel(self, b):
        ''' Mass flux density (sans sxb factor).
        Uses internal spatial variables.
        '''
        if(self.bessel_fm is None):
            self.bessel_fm = quad_vec(_flux_integrand, self.kmin, self.kmax,
                                      args=(self.b, self.J, self.S),
                                      )[0]
        return self.bessel_fm

    def _pressure_bessel_U(self):
        ''' The quantity pU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_pU is None):
            self.bessel_pU =quad_vec(_pressureU_integrand, self.kmin, self.kmax,
                                     args=(self.b, self.DU, self.cU),
                                     )[0]
        return self.bessel_pU

    def _pressure_bessel_T1(self):
        ''' The quantity pT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        '''
        if(self.bessel_pT1 is None):
            self.bessel_pT1 = quad_vec(_pressureT1_integrand_direct, self.kmin, self.kmax,
                                       args=(self.b, self.DT1, self.cT1),
                                       )[0]
        return self.bessel_pT1

    def _pressure_bessel_T2(self):
        ''' The quantity pT2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_pT2 is None):
            self.bessel_pT2 = quad_vec(_pressureT2_integrand, self.kmin, self.kmax,
                                       args=(self.b, self.DT2, self.cT2),
                                       )[0]
        return self.bessel_pT2

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_sU is None):
            self.bessel_sU = quad_vec(_shearU_integrand, self.kmin, self.kmax,
                                      args=(self.b, self.DU),
                                      )[0]
        return self.bessel_sU

    def _shear_bessel_T1(self, norder):
        ''' The quantity sT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        norder should be 0, 2 or 4.
        '''
        if(self.bessel_sT10 is None):
            self.bessel_sT10 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                        args=(self.b, self.DT1, 0),
                                        )[0]
        if(self.bessel_sT12 is None):
            self.bessel_sT12 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                        args=(self.b, self.DT1, 2),
                                        )[0]
        if(self.bessel_sT14 is None):
            self.bessel_sT14 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                        args=(self.b, self.DT1, 4),
                                        )[0]
        if(norder==0):
            return self.bessel_sT10
        if(norder==2):
            return self.bessel_sT12
        if(norder==4):
            return self.bessel_sT14
        raise ValueError("norder={:d} not recognized; should be 0, 2 or 4.".format(norder))

    def _shear_bessel_T2(self):
        ''' The quantity sT2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_sT2 is None):
            self.bessel_sT2 = quad_vec(_shearT2_integrand, self.kmin, self.kmax,
                                       args=(self.b, self.DT2),
                                       )[0]
        return self.bessel_sT2

    # 1D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These will become depreacted

    ##def mass_1D_U(self, b):
    ##    ''' Unpolarized part of mass density.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_massU_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.AU),
    ##                        )[0]
    ##    return integral

    ##def mass_1D_T(self, b):
    ##    ''' Tensor-polarized part of mass density.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_massT_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.AT),
    ##                        )[0]
    ##    return integral

    ##def momentum_1D(self, b):
    ##    ''' Momentum density (sans sxb factor).
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_momentum_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.J, self.S),
    ##                        )[0]
    ##    return integral

    ##def flux_1D(self, b):
    ##    ''' Mass flux density (sans sxb factor).
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_flux_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.J, self.S),
    ##                        )[0]
    ##    return integral

    ##def pressure_1D_U(self, b):
    ##    ''' The quantity pU, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_pressureU_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DU, self.cU),
    ##                        )[0]
    ##    return integral

    ##def pressure_1D_T1(self, b):
    ##    ''' The quantity pT1, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    This is the quantiy appearing in the Polyakov-Sun breakdown,
    ##    and isn't used numerically;
    ##    please use pressure_1D_T1_direct for the density instead.
    ##    '''
    ##    integral = quad_vec(_pressureT1_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DT1, self.cT1),
    ##                        )[0]
    ##    return integral

    ##def pressure_1D_T2(self, b):
    ##    ''' The quantity pT2, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_pressureT2_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DT2, self.cT2),
    ##                        )[0]
    ##    return integral

    ##def shear_1D_U(self, b):
    ##    ''' The quantity sU, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_shearU_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DU),
    ##                        )[0]
    ##    return integral

    ##def shear_1D_T1(self, b):
    ##    ''' The quantity sT1, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    This is the quantiy appearing in the Polyakov-Sun breakdown,
    ##    and isn't used numerically;
    ##    please use pressure_1D_T1_direct for the density instead.
    ##    '''
    ##    integral = quad_vec(_shearT1_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DT1),
    ##                        )[0]
    ##    return integral

    ##def shear_1D_T2(self, b):
    ##    ''' The quantity sT2, defined as a Bessel transform.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_shearT2_integrand, self.kmin, self.kmax,
    ##                        args=(b, self.DT2),
    ##                        )[0]
    ##    return integral

    # Direct Bessel transforms for T1 (don't need derivatives) ~~~~~~~~~~~~~~~~~

    ##def pressure_1D_T1_direct(self, b):
    ##    ''' The quantity pT1, defined as a Bessel transform.
    ##    This is from the alternate breakdown that avoids numerical derivatives.
    ##    b should be numpy.array in fm.
    ##    '''
    ##    integral = quad_vec(_pressureT1_integrand_direct, self.kmin, self.kmax,
    ##                        args=(b, self.DT1, self.cT1),
    ##                        )[0]
    ##    return integral

    ##def shear_1D_T1_direct(self, b, norder):
    ##    ''' The quantity sT1, defined as a Bessel transform.
    ##    This is from the alternate breakdown that avoids numerical derivatives.
    ##    b should be numpy.array in fm.
    ##    norder should be 0, 2 or 4.
    ##    '''
    ##    integral = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
    ##                        args=(b, self.DT1, norder),
    ##                        )[0]
    ##    return integral

    # 3D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NOTE: I will deprecate this, as they are inefficient

    ##def mass_3D_U(self, x, y, z):
    ##    ''' Three-dimensional mass density of the deuteron.
    ##    Unpolarized contribution.
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    return self.mass_1D_U(b)

    ##def mass_3D_T(self, x, y, z):
    ##    ''' Three-dimensional mass density of the deuteron.
    ##    Tensor-polarized contribution.
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    scalar = self.mass_1D_T(b)
    ##    Y2 = make_Y2(x, y, z)
    ##    rhoT = make_rhoT(x, y, z)
    ##    harmonics = np.einsum('xyzab,xyzab->xyz', Y2, rhoT)
    ##    return harmonics * scalar

    ##def momentum_3D(self, x, y, z):
    ##    ''' Three-dimensional momentum density of the deuteron.
    ##    Assumes spin up along the z axis.
    ##    Returns a 4D array, with dimensions (nx,ny,nz,3),
    ##    and the three components of the last being the
    ##    z,x,y components of the momentum (z=0 index).
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    scalar = self.momentum_1D(b)
    ##    s = make_zhat(x,y,z)
    ##    r = make_rhat(x,y,z)
    ##    sxb = cross_product(s,r)
    ##    result = np.einsum('xyz,xyzi->xyzi', scalar, sxb)
    ##    return result

    ##def flux_3D(self, x, y, z):
    ##    ''' Three-dimensional mass flux density of the deuteron.
    ##    Assumes spin up along the z axis.
    ##    Returns a 4D array, with dimensions (nx,ny,nz,3),
    ##    and the three components of the last being the
    ##    z,x,y components of the momentum (z=0 index).
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    scalar = self.flux_1D(b)
    ##    s = make_zhat(x,y,z)
    ##    r = make_rhat(x,y,z)
    ##    sxb = cross_product(s,r)
    ##    result = np.einsum('xyz,xyzi->xyzi', scalar, sxb)
    ##    return result

    ##def stress_3D_U(self, x, y, z):
    ##    ''' Three-dimensional stress tensor of the deuteron.
    ##    Unpolarized contribution.
    ##    Returns a 5D array, with dimensions (nx,ny,nz,3,3),
    ##    with the nine components of the last two being the nine components
    ##    of the stress tensor. z=0 index.
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    p = self.pressure_1D_U(b)
    ##    s = self.shear_1D_U(b)
    ##    dl = make_kronecker(x,y,z)
    ##    Y2 = make_Y2(x,y,z)
    ##    T0 = np.einsum('xyz,xyzij->xyzij', p, dl)
    ##    T2 = np.einsum('xyz,xyzij->xyzij', s, Y2)
    ##    return T0 + T2

    ##def stress_3D_T(self, x, y, z):
    ##    ''' Three-dimensional stress tensor of the deuteron.
    ##    Tesnor-polarized contribution.
    ##    Returns a 5D array, with dimensions (nx,ny,nz,3,3),
    ##    with the nine components of the last two being the nine components
    ##    of the stress tensor. z=0 index.
    ##    '''
    ##    return self.stress_3D_T1(x,y,z) + self.stress_3D_T2(x,y,z)

    ##def pr_3D_U(self, x, y, z):
    ##    ''' Three-dimensional radial pressure in the deuteron.
    ##    Unpolarized contribution.
    ##    Basically projects the result on stress_3D_U onto a unit radial vector.
    ##    '''
    ##    T = self.stress_3D_U(x, y, z)
    ##    r = make_rhat(x, y, z)
    ##    pr = np.einsum('xyzij,xyzi,xyzj->xyz', T, r, r)
    ##    return pr

    ##def pr_3D_T(self, x, y, z):
    ##    ''' Three-dimensional radial pressure in the deuteron.
    ##    Tensor-polarized contribution.
    ##    Basically projects the result on stress_3D_T onto a unit radial vector.
    ##    '''
    ##    T = self.stress_3D_T(x, y, z)
    ##    r = make_rhat(x, y, z)
    ##    pr = np.einsum('xyzij,xyzi,xyzj->xyz', T, r, r)
    ##    return pr

    ##def piso_3D_U(self, x, y, z):
    ##    ''' Three-dimensional isotropic pressure in the deuteron.
    ##    Unpolarized contribution.
    ##    Basically projects the result on stress_3D_U onto kronecker/3.
    ##    '''
    ##    T = self.stress_3D_U(x, y, z)
    ##    dl = make_kronecker(x, y, z)
    ##    pressure = np.einsum('xyzij,xyzij->xyz', T, dl) / 3
    ##    return pressure

    ##def piso_3D_T(self, x, y, z):
    ##    ''' Three-dimensional isotropic pressure in the deuteron.
    ##    Tensor-polarized contribution.
    ##    Basically projects the result on stress_3D_T onto onto kronecker/3.
    ##    '''
    ##    T = self.stress_3D_T(x, y, z)
    ##    dl = make_kronecker(x, y, z)
    ##    pressure = np.einsum('xyzij,xyzij->xyz', T, dl) / 3
    ##    return pressure

    # Separated T1 and T2 contributions to tensor polarized stress ~~~~~~~~~~~~~

    ##def stress_3D_T1(self, x, y, z):
    ##    ''' Three-dimensional stress tensor of the deuteron.
    ##    Tesnor-polarized contribution, from DT1 and cT1.
    ##    Returns a 5D array, with dimensions (nx,ny,nz,3,3),
    ##    with the nine components of the last two being the nine components
    ##    of the stress tensor. z=0 index.
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    # Direct integrals for pieces of this stress
    ##    p = quad_vec(_pressureT1_integrand_direct, self.kmin, self.kmax,
    ##                 args=(b, self.DT1, self.cT1),
    ##                 )[0]
    ##    s0 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
    ##                  args=(b, self.DT1, 0),
    ##                  )[0]
    ##    s2 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
    ##                  args=(b, self.DT1, 2),
    ##                  )[0]
    ##    s4 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
    ##                  args=(b, self.DT1, 4),
    ##                  )[0]
    ##    # Rest of the calculation
    ##    dl = make_kronecker(x,y,z)
    ##    Y2 = make_Y2(x,y,z)
    ##    Y4 = make_Y4(x,y,z)
    ##    X0 = make_X0(x,y,z)
    ##    X2 = make_X2(x,y,z)
    ##    T0  = np.einsum('xyz,xyzab,xyzij->xyzijab', p, Y2, dl)
    ##    T2a = np.einsum('xyz,xyzijab->xyzijab', s0, X0)
    ##    T2b = np.einsum('xyz,xyzijab->xyzijab', s2, X2)
    ##    T2c = np.einsum('xyz,xyzijab->xyzijab', s4, Y4)
    ##    rhoT = make_rhoT(x, y, z)
    ##    T = np.einsum('xyzijab,xyzab->xyzij', T0+T2a+T2b+T2c, rhoT)
    ##    return T

    ##def stress_3D_T2(self, x, y, z):
    ##    ''' Three-dimensional stress tensor of the deuteron.
    ##    Tesnor-polarized contribution, from DT2 and cT2.
    ##    Returns a 5D array, with dimensions (nx,ny,nz,3,3),
    ##    with the nine components of the last two being the nine components
    ##    of the stress tensor. z=0 index.
    ##    '''
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    b = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    p = self.pressure_1D_T2(b)
    ##    s = self.shear_1D_T2(b)
    ##    dl = make_kronecker(x,y,z)
    ##    Y2 = make_Y2(x,y,z)
    ##    Q = make_Q(x,y,z)
    ##    QY2 = 2*(
    ##            np.einsum('xyzilab,xyzlj->xyzijab', Q, Y2)
    ##            +
    ##            np.einsum('xyzjlab,xyzli->xyzijab', Q, Y2)
    ##            -
    ##            np.einsum('xyzlkab,xyzlk,xyzij->xyzijab', Q, Y2, dl)
    ##            )
    ##    T0 = np.einsum('xyz,xyzijab->xyzijab', p, Q)
    ##    T2 = np.einsum('xyz,xyzijab->xyzijab', s, QY2)
    ##    rhoT = make_rhoT(x, y, z)
    ##    T = np.einsum('xyzijab,xyzab->xyzij', T0+T2, rhoT)
    ##    return T

    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _initialize_space(self):
        ''' Initialize 3D grids of b, theta and phi variables. '''
        b = np.linspace(-self.bmax, self.bmax, self.nb)
        x, y, z = np.meshgrid(b, b, b, indexing='ij')
        self.b = np.sqrt(x**2 + y**2 + z**2)
        self.theta = np.atan(np.sqrt(x**2+y**2)/z)
        self.phi = np.atan(y/x)
        self.x = b # a 1D array to grab for making plots
        return

    def _initialize_bessel(self):
        ''' Ensure variables for cached Bessel transforms exist.
        Initialize them to 0.
        '''
        self.bessel_aU   = None
        self.bessel_aT   = None
        self.bessel_k    = None
        self.bessel_fm   = None
        self.bessel_pU   = None
        self.bessel_sU   = None
        self.bessel_pT1  = None
        self.bessel_sT10 = None
        self.bessel_sT12 = None
        self.bessel_sT14 = None
        self.bessel_pT2  = None
        self.bessel_sT2  = None
        return

    ##def _xyz2spherical(self, x, y, z):
    ##    # TODO: deprecate
    ##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
    ##    r_ = np.sqrt(x_**2 + y_**2 + z_**2)
    ##    theta_ = np.atan(np.sqrt(x_**2+y_**2)/z_)
    ##    phi_ = np.atan(y_/x_)
    ##    return r_, theta_, phi_

    def _cache_path(self):
        filename = "mff_table_{}_{}_{:d}_{:.2e}_{:.2e}".format(
                self.wf, self.nff, self.nk, self.kmin, self.kmax
                )
        path = Path(__file__).parent / 'cache/{}.csv'.format(filename)
        return path

    def _init_mff_table(self, save_table=False):
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
        if(save_table):
            df = pd.DataFrame(data={
                'k'   : k,
                'AU'  : AU,
                'AT'  : AT,
                'DU'  : DU,
                'DT1' : DT1,
                'DT2' : DT2,
                'cU'  : cU,
                'cT1' : cT1,
                'cT2' : cT2,
                'J'   : J,
                'S'   : S
                })
            path = self._cache_path()
            df.to_csv(path, index=None)
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

    def _load_mff_table(self, filename):
        df = pd.read_csv(filename)
        k = df['k'].to_numpy()
        AU  = df['AU'].to_numpy()
        AT  = df['AT'].to_numpy()
        DU  = df['DU'].to_numpy()
        DT1 = df['DT1'].to_numpy()
        DT2 = df['DT2'].to_numpy()
        cU  = df['cU'].to_numpy()
        cT1 = df['cT1'].to_numpy()
        cT2 = df['cT2'].to_numpy()
        J   = df['J'].to_numpy()
        S   = df['S'].to_numpy()
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

    def _pol_error(self,pol):
        raise ValueError(
                "pol={} not recognized; use 'U', 'T', 1, 0 or -1.".format(pol)
                )

    def _pol_error_eigen(self,pol):
        raise ValueError(
                "pol={} invalid for eigenpressures; use 1, 0 or -1.".format(pol)
                )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to make vectors, harmonic tensors, etc.

##def make_rhat(x, y, z):
##    # TODO: docstring
##    eps = 1e-9 # to regulate division by zero
##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
##    r_ = np.sqrt(x_**2 + y_**2 + z_**2 + eps)
##    rhat = np.zeros(x_.shape+(3,))
##    rhat[...,0] = z_/r_
##    rhat[...,1] = x_/r_
##    rhat[...,2] = y_/r_
##    return rhat
##
##def make_zhat(x, y, z):
##    # TODO: docstring
##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
##    zhat = np.zeros(x_.shape+(3,))
##    zhat[...,0] = 1
##    return zhat
##
##def make_phihat(x,y,z):
##    # TODO: docstring
##    eps = 1e-9 # to regulate division by zero
##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
##    rho_ = np.sqrt(x_**2 + y_**2 + eps)
##    phihat = np.zeros(x_.shape+(3,))
##    phihat[...,0] = 0
##    phihat[...,1] = -y_/rho_
##    phihat[...,2] = x_/rho_
##    return phihat
##
##def make_thetahat(x,y,z):
##    # TODO: docstring
##    eps = 1e-9 # to regulate division by zero
##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
##    rho_ = np.sqrt(x_**2 + y_**2 + eps)
##    r_ = np.sqrt(x_**2 + y_**2 + z_**2 + eps)
##    thetahat = np.zeros(x_.shape+(3,))
##    thetahat[...,0] = -rho_/r_
##    thetahat[...,1] = x_*z_/(rho_*r_)
##    thetahat[...,2] = y_*z_/(rho_*r_)
##    return thetahat
##
##def cross_product(A,B):
##    # TODO: docstring
##    C = np.zeros(A.shape)
##    C[...,0] = A[...,1]*B[...,2] - A[...,2]*B[...,1]
##    C[...,1] = A[...,2]*B[...,0] - A[...,0]*B[...,2]
##    C[...,2] = A[...,0]*B[...,1] - A[...,1]*B[...,0]
##    return C
##
##def make_kronecker(x, y, z):
##    # TODO: docstring
##    x_, y_, z_ = np.meshgrid(x, y, z, indexing='ij')
##    kronecker = np.zeros(x_.shape+(3,3))
##    kronecker[...,0,0] = 1
##    kronecker[...,1,1] = 1
##    kronecker[...,2,2] = 1
##    return kronecker
##
##def make_Q(x, y, z):
##    # TODO: docstring
##    d = make_kronecker(x,y,z)
##    dd1 = np.einsum('xyzij,xyzab->xyzijab', d, d)
##    dd2 = np.einsum('xyzai,xyzbj->xyzijab', d, d)
##    dd3 = np.einsum('xyzaj,xyzbi->xyzijab', d, d)
##    return (dd2+dd3)/2 - dd1/3
##
##def make_Y1(x, y, z):
##    # TODO: docstring
##    return make_rhat(x, y, z)
##
##def make_Y2(x, y, z):
##    # TODO: docstring
##    r = make_rhat(x,y,z)
##    d = make_kronecker(x,y,z)
##    rr = np.einsum('xyzi,xyzj->xyzij', r, r)
##    return rr - d/3
##
##def make_Y3(x, y, z):
##    # TODO: docstring
##    r = make_rhat(x,y,z)
##    d = make_kronecker(x,y,z)
##    rrr = np.einsum('xyzi,xyzj,xyzk->xyzijk', r, r, r)
##    dr1 = np.einsum('xyzij,xyzk->xyzijk', d, r)
##    dr2 = np.einsum('xyzki,xyzj->xyzijk', d, r)
##    dr3 = np.einsum('xyzjk,xyzi->xyzijk', d, r)
##    return rrr - (dr1+dr2+dr3)/3
##
##def make_Y4(x, y, z):
##    # TODO: docstring
##    r = make_rhat(x,y,z)
##    d = make_kronecker(x,y,z)
##    rrrr = np.einsum('xyzi,xyzj,xyzk,xyzl->xyzijkl', r, r, r, r)
##    drr1 = np.einsum('xyzij,xyzk,xyzl->xyzijkl', d, r, r)
##    drr2 = np.einsum('xyzkl,xyzi,xyzj->xyzijkl', d, r, r)
##    drr3 = np.einsum('xyzik,xyzj,xyzl->xyzijkl', d, r, r)
##    drr4 = np.einsum('xyzjk,xyzi,xyzl->xyzijkl', d, r, r)
##    drr5 = np.einsum('xyzil,xyzj,xyzk->xyzijkl', d, r, r)
##    drr6 = np.einsum('xyzjl,xyzi,xyzk->xyzijkl', d, r, r)
##    dd1 = np.einsum('xyzij,xyzkl->xyzijkl', d, d)
##    dd2 = np.einsum('xyzik,xyzjl->xyzijkl', d, d)
##    dd3 = np.einsum('xyzil,xyzkj->xyzijkl', d, d)
##    return (
##            rrrr
##            - (drr1 + drr2 + drr3 + drr4 + drr5 + drr6)/7
##            + (dd1 + dd2 + dd3)/35
##            )
##
### Peculiar tensors appearing in tensor-polarized stresses ~~~~~~~~~~~~~~~~~~~~~~
##
##def make_X2(x, y, z):
##    # TODO: docstring
##    Y2 = make_Y2(x, y, z)
##    dl = make_kronecker(x, y, z)
##    ijab = np.einsum('xyzij,xyzab->xyzijab', Y2, dl)
##    abij = np.einsum('xyzab,xyzij->xyzijab', Y2, dl)
##    aibj = np.einsum('xyzai,xyzbj->xyzijab', Y2, dl)
##    ajbi = np.einsum('xyzaj,xyzbi->xyzijab', Y2, dl)
##    bjai = np.einsum('xyzbj,xyzai->xyzijab', Y2, dl)
##    biaj = np.einsum('xyzbi,xyzaj->xyzijab', Y2, dl)
##    term1 = (ijab + abij + aibj + ajbi + bjai + biaj)/7
##    term2 = -(ijab + abij)/3
##    return term1 + term2
##
##def make_X0(x, y, z):
##    # TODO: docstring
##    dl = make_kronecker(x, y, z)
##    ijab = np.einsum('xyzij,xyzab->xyzijab', dl, dl)
##    aibj = np.einsum('xyzai,xyzbj->xyzijab', dl, dl)
##    ajbi = np.einsum('xyzaj,xyzbi->xyzijab', dl, dl)
##    term1 = (ijab + aibj + ajbi)/15
##    term2 = -ijab/9
##    return term1 + term2
##
### Spin density matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
##def make_rhoT(x, y, z):
##    # TODO: docstring
##    zh = make_zhat(x,y,z)
##    dl = make_kronecker(x,y,z)
##    zz = np.einsum('xyzi,xyzj->xyzij', zh, zh)
##    return 3/2*zz - dl/2

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

def _pressureU_integrand(k, b, DU, cU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DU(k) + 2*mN*cU(k)
    return common * unique * bessel * form

def _pressureT1_integrand(k, b, DT1, cT1):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def _pressureT2_integrand(k, b, DT2, cT2):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(12*mN)*DT2(k) + 2*mN*cT2(k)
    return common * unique * bessel * form

def _shearU_integrand(k, b, DU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DU(k)
    return common * unique * bessel * form

def _shearT1_integrand(k, b, DT1):
    # NOTE: this is the Polyakov-Sun density, which must be differentiated
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

def _shearT2_integrand(k, b, DT2):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*DT2(k)
    return common * unique * bessel * form

# T1 stress integrals for direct use (no differentiation) ~~~~~~~~~~~~~~~~~~~~~~

def _pressureT1_integrand_direct(k, b, DT1, cT1):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = +k**2/(8*mN**2)
    bessel = jn(2, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def _shearT1_integrand_direct(k, b, DT1, norder):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = (-1)**(norder//2) * k**2/(8*mN**2)
    bessel = jn(norder, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

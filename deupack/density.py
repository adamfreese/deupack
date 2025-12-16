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
from . import emtff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density class

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
                 nb=101,
                 bmax=2,    # fm
                 kmin=1e-6, # GeV
                 kmax=10    # GeV
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
        if(path.is_file()):
            self._load_emtff_table(path)
        else:
            # if not found, make one
            self._init_emtff_table(save_table=True)
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

    def momentum_density(self, pol='U'):
        ''' Phi component of the momentum density, in GeV/fm**3. '''
        if(pol=='U' or pol=='T' or pol==0):
            return np.zeros(self.b.shape)
        elif(pol==1 or pol==-1):
            b_dep = self._momentum_bessel()
            theta_dep = np.sin(self.theta)
            return b_dep * theta_dep * pol
        else:
            self._pol_error(pol)

    def flux_density(self, pol='U'):
        ''' Phi component of the mass flux density, in GeV/fm**3. '''
        if(pol=='U' or pol=='T' or pol==0):
            return np.zeros(self.b.shape)
        elif(pol==1 or pol==-1):
            b_dep = self._flux_bessel()
            theta_dep = np.sin(self.theta)
            return b_dep * theta_dep * pol
        else:
            self._pol_error(pol)

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

    def symmetric_shear(self, pol='U'):
        ''' Symmetric shear in the r-theta directions, in GeV/fm**3. '''
        theta_dep = np.sin(self.theta) * np.cos(self.theta)
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

    def torsion_shear(self, pol='U'):
        ''' Antisymmetric shear in the r-theta direction, in GeV/fm**3. '''
        theta_dep = np.sin(self.theta) * np.cos(self.theta)
        b_dep = 3*self._shear_bessel_A()
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
        #return 0.5*(pr + pt + (pr-pt)*np.sqrt(1+4*s**2/(pr-pt)**2))

    def pseudolateral_pressure(self, pol='U'):
        ''' Eigenpressure closest to the lateral direction, in GeV/fm**3. '''
        pr = self.radial_pressure(pol=pol)
        pt = self.lateral_pressure(pol=pol)
        s = self.symmetric_shear()
        return 0.5*(pr+pt - np.sqrt((pr-pt)**2+4*s**2))
        #return 0.5*(pr + pt - (pr-pt)*np.sqrt(1+4*s**2/(pr-pt)**2))

    # Unit eigenvectors of the symmetric stress tensor ~~~~~~~~~~~~~~~~~~~~~~~~~

    def e_plus(self, pol='U'):
        ''' Returns three 3D numpy arrays, with the Cartesian x, y and z
        components of the pseudoradial eigenvector of the symmetric stress
        tensor.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.lateral_pressure(pol=pol)
        s  = self.symmetric_shear( pol=pol)
        sgn = np.sign(s)
        R  = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Z = R*np.cos(self.theta) - sgn*Th*np.sin(self.theta)
        X = (R*np.sin(self.theta) + sgn*Th*np.cos(self.theta))*np.cos(self.phi)
        Y = (R*np.sin(self.theta) + sgn*Th*np.cos(self.theta))*np.sin(self.phi)
        return X, Y, Z

    def e_minus(self, pol='U'):
        ''' Returns three 3D numpy arrays, with the Cartesian x, y and z
        components of the pseudoradial eigenvector of the symmetric stress
        tensor.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.lateral_pressure(pol=pol)
        s  = self.symmetric_shear( pol=pol)
        sgn = np.sign(s)
        R  = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Z = R*np.cos(self.theta) + sgn*Th*np.sin(self.theta)
        X = (R*np.sin(self.theta) - sgn*Th*np.cos(self.theta))*np.cos(self.phi)
        Y = (R*np.sin(self.theta) - sgn*Th*np.cos(self.theta))*np.sin(self.phi)
        return X, Y, Z

    # Bessel transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _mass_bessel_U(self):
        ''' Unpolarized part of mass density; b dependence.
        Uses internal spatial variables.
        '''
        if(self.bessel_aU is None):
            path = self._cache_path_bessel('aU')
            if(path.is_file()):
                self.bessel_aU = np.load(path)
            else:
                self.bessel_aU = quad_vec(_massU_integrand, self.kmin, self.kmax,
                                          args=(self.b, self.AU),
                                          workers=8)[0]
            np.save(path, self.bessel_aU)
        return self.bessel_aU

    def _mass_bessel_T(self):
        ''' Tesnor-polarized part of mass density; b dependence.
        Uses internal spatial variables.
        '''
        if(self.bessel_aT is None):
            path = self._cache_path_bessel('aT')
            if(path.is_file()):
                self.bessel_aT = np.load(path)
            else:
                self.bessel_aT = quad_vec(_massT_integrand, self.kmin, self.kmax,
                                          args=(self.b, self.AT),
                                          workers=8)[0]
            np.save(path, self.bessel_aT)
        return self.bessel_aT

    def _momentum_bessel(self):
        ''' Momentum density (sans sxb factor).
        Uses internal spatial variables.
        '''
        if(self.bessel_k is None):
            path = self._cache_path_bessel('k')
            if(path.is_file()):
                self.bessel_k = np.load(path)
            else:
                self.bessel_k = quad_vec(_momentum_integrand, self.kmin, self.kmax,
                                         args=(self.b, self.J, self.S),
                                         workers=8)[0]
                np.save(path, self.bessel_k)
        return self.bessel_k

    def _flux_bessel(self):
        ''' Mass flux density (sans sxb factor).
        Uses internal spatial variables.
        '''
        if(self.bessel_fm is None):
            path = self._cache_path_bessel('fm')
            if(path.is_file()):
                self.bessel_fm = np.load(path)
            else:
                self.bessel_fm = quad_vec(_flux_integrand, self.kmin, self.kmax,
                                          args=(self.b, self.J, self.S),
                                          workers=8)[0]
                np.save(path, self.bessel_fm)
        return self.bessel_fm

    def _pressure_bessel_U(self):
        ''' The quantity pU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_pU is None):
            path = self._cache_path_bessel('pU')
            if(path.is_file()):
                self.bessel_pU = np.load(path)
            else:
                self.bessel_pU =quad_vec(_pressureU_integrand, self.kmin, self.kmax,
                                         args=(self.b, self.DU, self.cU),
                                         workers=8)[0]
                np.save(path, self.bessel_pU)
        return self.bessel_pU

    def _pressure_bessel_T1(self):
        ''' The quantity pT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        '''
        if(self.bessel_pT1 is None):
            path = self._cache_path_bessel('pT1')
            if(path.is_file()):
                self.bessel_pT1 = np.load(path)
            else:
                self.bessel_pT1 = quad_vec(_pressureT1_integrand_direct, self.kmin, self.kmax,
                                           args=(self.b, self.DT1, self.cT1),
                                           workers=8)[0]
                np.save(path, self.bessel_pT1)
        return self.bessel_pT1

    def _pressure_bessel_T2(self):
        ''' The quantity pT2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_pT2 is None):
            path = self._cache_path_bessel('pT2')
            if(path.is_file()):
                self.bessel_pT2 = np.load(path)
            else:
                self.bessel_pT2 = quad_vec(_pressureT2_integrand, self.kmin, self.kmax,
                                           args=(self.b, self.DT2, self.cT2),
                                           workers=8)[0]
                np.save(path, self.bessel_pT2)
        return self.bessel_pT2

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_sU is None):
            path = self._cache_path_bessel('sU')
            if(path.is_file()):
                self.bessel_sU = np.load(path)
            else:
                self.bessel_sU = quad_vec(_shearU_integrand, self.kmin, self.kmax,
                                          args=(self.b, self.DU),
                                          workers=8)[0]
                np.save(path, self.bessel_sU)
        return self.bessel_sU

    def _shear_bessel_T1(self, norder):
        ''' The quantity sT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        norder should be 0, 2 or 4.
        '''
        if(self.bessel_sT10 is None):
            path0 = self._cache_path_bessel('sT10')
            if(path0.is_file()):
                self.bessel_sT10 = np.load(path0)
            else:
                self.bessel_sT10 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                            args=(self.b, self.DT1, 0),
                                            workers=8)[0]
                np.save(path0, self.bessel_sT10)
        if(self.bessel_sT12 is None):
            path2 = self._cache_path_bessel('sT12')
            if(path2.is_file()):
                self.bessel_sT12 = np.load(path2)
            else:
                self.bessel_sT12 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                            args=(self.b, self.DT1, 2),
                                            workers=8)[0]
                np.save(path2, self.bessel_sT12)
        if(self.bessel_sT14 is None):
            path4 = self._cache_path_bessel('sT14')
            if(path4.is_file()):
                self.bessel_sT14 = np.load(path4)
            else:
                self.bessel_sT14 = quad_vec(_shearT1_integrand_direct, self.kmin, self.kmax,
                                            args=(self.b, self.DT1, 4),
                                            workers=8)[0]
                np.save(path4, self.bessel_sT14)
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
            path = self._cache_path_bessel('sT2')
            if(path.is_file()):
                self.bessel_sT2 = np.load(path)
            else:
                self.bessel_sT2 = quad_vec(_shearT2_integrand, self.kmin, self.kmax,
                                           args=(self.b, self.DT2),
                                           workers=8)[0]
                np.save(path, self.bessel_sT2)
        return self.bessel_sT2

    def _shear_bessel_A(self):
        ''' The quantity sA, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        if(self.bessel_sA is None):
            path = self._cache_path_bessel('sA')
            if(path.is_file()):
                self.bessel_sA = np.load(path)
            else:
                self.bessel_sA = quad_vec(_shearA_integrand, self.kmin, self.kmax,
                                           args=(self.b, self.sbar),
                                           workers=8)[0]
                np.save(path, self.bessel_sA)
        return self.bessel_sA

    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _initialize_space(self):
        ''' Initialize 3D grids of b, theta and phi variables. '''
        b = np.linspace(-self.bmax, self.bmax, self.nb)
        x, y, z = np.meshgrid(b, b, b, indexing='ij')
        self.b = np.sqrt(x**2 + y**2 + z**2)
        self.theta = np.atan2(np.sqrt(x**2+y**2 + 1e-9), z)
        self.phi = np.atan2(y, x)
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
        self.bessel_sA   = None
        return

    def _cache_path(self):
        filename = "emtff_table_{}_{}_{:d}_{:.2e}_{:.2e}".format(
                self.wf, self.nff, self.nk, self.kmin, self.kmax
                )
        path = Path(__file__).parent / 'cache/{}.csv'.format(filename)
        return path

    def _cache_path_bessel(self, name):
        filename = "bessel_{}_table_{}_{}_{:d}_{:.2e}".format(
                name, self.wf, self.nff, self.nb, self.bmax,
                )
        path = Path(__file__).parent / 'cache/{}.npy'.format(filename)
        return path

    def _init_emtff_table(self, save_table=False):
        k    = np.geomspace(self.kmin, self.kmax, self.nk)
        AU   = emtff.AU(  k, wf=self.wf, nff=self.nff)
        AT   = emtff.AT(  k, wf=self.wf, nff=self.nff)
        DU   = emtff.DU(  k, wf=self.wf, nff=self.nff)
        DT1  = emtff.DT1( k, wf=self.wf, nff=self.nff)
        DT2  = emtff.DT2( k, wf=self.wf, nff=self.nff)
        cU   = emtff.cU(  k, wf=self.wf, nff=self.nff)
        cT1  = emtff.cT1( k, wf=self.wf, nff=self.nff)
        cT2  = emtff.cT2( k, wf=self.wf, nff=self.nff)
        J    = emtff.J(   k, wf=self.wf, nff=self.nff)
        S    = emtff.S(   k, wf=self.wf, nff=self.nff)
        sbar = emtff.sbar(k, wf=self.wf, nff=self.nff)
        if(save_table):
            df = pd.DataFrame(data={
                'k'    : k,
                'AU'   : AU,
                'AT'   : AT,
                'DU'   : DU,
                'DT1'  : DT1,
                'DT2'  : DT2,
                'cU'   : cU,
                'cT1'  : cT1,
                'cT2'  : cT2,
                'J'    : J,
                'S'    : S,
                'sbar' : sbar
                })
            path = self._cache_path()
            df.to_csv(path, index=None)
        self.AU   = CubicSpline(k, AU)
        self.AT   = CubicSpline(k, AT)
        self.DU   = CubicSpline(k, DU)
        self.DT1  = CubicSpline(k, DT1)
        self.DT2  = CubicSpline(k, DT2)
        self.cU   = CubicSpline(k, cU)
        self.cT1  = CubicSpline(k, cT1)
        self.cT2  = CubicSpline(k, cT2)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        self.sbar = CubicSpline(k, sbar)
        return

    def _load_emtff_table(self, filename):
        df   = pd.read_csv(filename)
        k    = df['k'].to_numpy()
        AU   = df['AU'].to_numpy()
        AT   = df['AT'].to_numpy()
        DU   = df['DU'].to_numpy()
        DT1  = df['DT1'].to_numpy()
        DT2  = df['DT2'].to_numpy()
        cU   = df['cU'].to_numpy()
        cT1  = df['cT1'].to_numpy()
        cT2  = df['cT2'].to_numpy()
        J    = df['J'].to_numpy()
        S    = df['S'].to_numpy()
        sbar = df['sbar'].to_numpy()
        self.AU   = CubicSpline(k, AU)
        self.AT   = CubicSpline(k, AT)
        self.DU   = CubicSpline(k, DU)
        self.DT1  = CubicSpline(k, DT1)
        self.DT2  = CubicSpline(k, DT2)
        self.cU   = CubicSpline(k, cU)
        self.cT1  = CubicSpline(k, cT1)
        self.cT2  = CubicSpline(k, cT2)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        self.sbar = CubicSpline(k, sbar)
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

def _shearA_integrand(k, b, sbar):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(8*mN)*sbar(k)
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

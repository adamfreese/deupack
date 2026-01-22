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
    with different EMT-FFs in their cache. Additionally, the density functions
    can themselves be cached on a per-object basis to reduce computation time
    when creating 3D density plots.
    '''

    def __init__(self,
                 wf=emtff.wf_default,
                 nff='ba',
                 nk=600,
                 nb=101,
                 bmax=2,    # fm
                 kmin=1e-6, # GeV
                 kmax=20    # GeV
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
            return self.mass_density(pol='U') - 1/3*self.mass_density(pol='T')
        elif(pol==1 or pol==-1):
            return self.mass_density(pol='U') + 1/6*self.mass_density(pol='T')
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
                    2*self._pressure_bessel_T1()
                    -
                    4/15*self._shear_bessel_T1(0)
                    -
                    8/21*self._shear_bessel_T1(2)
                    -
                    24/35*self._shear_bessel_T1(4)
                    +
                    2*self._pressure_bessel_T2()
                    +
                    4/3*self._shear_bessel_T2()
                    )
            theta_dep = 3/2*np.cos(self.theta)**2 - 1/2
            return b_dep*theta_dep
        elif(pol==0):
            return self.radial_pressure(pol='U') - 1/3*self.radial_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.radial_pressure(pol='U') + 1/6*self.radial_pressure(pol='T')
        else:
            self._pol_error(pol)

    def polar_pressure(self, pol='U'):
        ''' Lateral pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() - 1/3*self._shear_bessel_U()
        elif(pol=='T'):
            b_dep1 = (
                    2*self._pressure_bessel_T1()
                    +
                    4/15*self._shear_bessel_T1(0)
                    +
                    2/5*self._shear_bessel_T1(4)
                    -
                    2*self._pressure_bessel_T2()
                    -
                    4/3*self._shear_bessel_T2()
                    )
            b_dep0 = (
                    -
                    2/15*self._shear_bessel_T1(0)
                    +
                    4/21*self._shear_bessel_T1(2)
                    -
                    2/35*self._shear_bessel_T1(4)
                    +
                    self._pressure_bessel_T2()
                    -
                    4/3*self._shear_bessel_T2()
                    )
            theta_dep1 = 3/2*np.cos(self.theta)**2 - 1/2
            return theta_dep1*b_dep1 + b_dep0
        elif(pol==0):
            return self.polar_pressure(pol='U') - 1/3*self.polar_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.polar_pressure(pol='U') + 1/6*self.polar_pressure(pol='T')
        else:
            self._pol_error(pol)

    def azimuthal_pressure(self, pol='U'):
        ''' Lateral pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() - 1/3*self._shear_bessel_U()
        elif(pol=='T'):
            b_dep1 = (
                    2*self._pressure_bessel_T1()
                    +
                    8/21*self._shear_bessel_T1(2)
                    +
                    2/7*self._shear_bessel_T1(4)
                    -
                    4*self._shear_bessel_T2()
                    )
            b_dep0 = (
                    2/15*self._shear_bessel_T1(0)
                    -
                    4/21*self._shear_bessel_T1(2)
                    +
                    2/35*self._shear_bessel_T1(4)
                    -
                    self._pressure_bessel_T2()
                    +
                    4/3*self._shear_bessel_T2()
                    )
            theta_dep1 = 3/2*np.cos(self.theta)**2 - 1/2
            return theta_dep1*b_dep1 + b_dep0
        elif(pol==0):
            return self.azimuthal_pressure(pol='U') - 1/3*self.azimuthal_pressure(pol='T')
        elif(pol==1 or pol==-1):
            return self.azimuthal_pressure(pol='U') + 1/6*self.azimuthal_pressure(pol='T')
        else:
            self._pol_error(pol)

    def symmetric_shear(self, pol='U'):
        ''' Symmetric shear in the r-theta directions, in GeV/fm**3. '''
        theta_dep = np.sin(self.theta) * np.cos(self.theta)
        b_dep = (
                2/5*self._shear_bessel_T1(0)
                +
                2/7*self._shear_bessel_T1(2)
                -
                24/35*self._shear_bessel_T1(4)
                -
                3*self._pressure_bessel_T2()
                -
                2*self._shear_bessel_T2()
                )
        shear = theta_dep*b_dep
        if(pol=='U'):
            shear *= 0
        elif(pol=='T'):
            pass
        elif(pol==0):
            shear *= -1/3
        elif(pol==1 or pol==-1):
            shear *= 1/6
        else:
            self._pol_error(pol)
        return shear

    def torsion_shear(self, pol='U'):
        ''' Antisymmetric shear in the r-theta direction, in GeV/fm**3. '''
        theta_dep = np.sin(self.theta) * np.cos(self.theta)
        b_dep = 6*self._shear_bessel_A()
        shear = theta_dep*b_dep
        if(pol=='U'):
            shear *= 0
        elif(pol=='T'):
            pass
        elif(pol==0):
            shear *= -1/3
        elif(pol==1 or pol==-1):
            shear *= 1/6
        else:
            self._pol_error(pol)
        return shear

    def isoradial_pressure(self, pol='U'):
        ''' Principal stress closest to the radial direction, in GeV/fm**3. '''
        pr = self.radial_pressure(pol=pol)
        pt = self.polar_pressure(pol=pol)
        s = self.symmetric_shear(pol=pol)
        return 0.5*(pr+pt + np.sqrt((pr-pt)**2+4*s**2))

    def isopolar_pressure(self, pol='U'):
        ''' Principal stress closest to the polar direction, in GeV/fm**3. '''
        pr = self.radial_pressure(pol=pol)
        pt = self.polar_pressure(pol=pol)
        s = self.symmetric_shear()
        return 0.5*(pr+pt - np.sqrt((pr-pt)**2+4*s**2))

    def radial_force(self, pol='U'):
        ''' Radial force density, in GeV/fm**4. '''
        if(pol=='U'):
            return self._force_bessel_0()
        elif(pol=='T'):
            b_dep = self._force_bessel_2() + 3/5*self._force_bessel_3()
            theta_dep = (3*np.cos(self.theta)**2 - 1)
            return b_dep*theta_dep
        elif(pol==0):
            return self.radial_force(pol='U') - 1/3*self.radial_force(pol='T')
        elif(pol==1 or pol==-1):
            return self.radial_force(pol='U') + 1/6*self.radial_force(pol='T')
        else:
            self._pol_error(pol)

    def polar_force(self, pol='U'):
        ''' Polar force density, in GeV/fm**4. '''
        theta_dep = -3 * np.sin(self.theta) * np.cos(self.theta)
        b_dep = self._force_bessel_2() - 2/5*self._force_bessel_3()
        force = theta_dep*b_dep
        if(pol=='U'):
            force *= 0
        elif(pol=='T'):
            pass
        elif(pol==0):
            force *= -1/3
        elif(pol==1 or pol==-1):
            force *= 1/6
        else:
            self._pol_error(pol)
        return force

    # Principal axes of the symmetric stress tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def e_plus(self, pol='U'):
        ''' Returns three 3D numpy arrays, with the Cartesian x, y and z
        components of the isoradial principal axis.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.polar_pressure(pol=pol)
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
        components of the isopolar principal axis.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.polar_pressure(pol=pol)
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
        return self._bessel_array('aU', _massU_integrand, self.AU)

    def _mass_bessel_T(self):
        ''' Tensor-polarized part of mass density; b dependence.
        Uses internal spatial variables.
        '''
        return self._bessel_array('aT', _massT_integrand, self.AT)

    def _momentum_bessel(self):
        ''' Momentum density (sans sxb factor).
        Uses internal spatial variables.
        '''
        return self._bessel_array('k', _momentum_integrand, self.J, self.S)

    def _flux_bessel(self):
        ''' Mass flux density (sans sxb factor).
        Uses internal spatial variables.
        '''
        return self._bessel_array('fm', _flux_integrand, self.J, self.S)

    def _pressure_bessel_U(self):
        ''' The quantity pU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pU', _pressureU_integrand, self.DU, self.cU)

    def _pressure_bessel_T1(self):
        ''' The quantity pT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pT1', _pressureT1_integrand_direct, self.DT1, self.cT1)

    def _pressure_bessel_T2(self):
        ''' The quantity pT2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pT2', _pressureT2_integrand, self.DT2, self.cT2)

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sU', _shearU_integrand, self.DU)

    def _shear_bessel_T1(self, norder):
        ''' The quantity sT1-tilde, defined as a Bessel transform.
        This is from the alternate breakdown that avoids numerical derivatives.
        Uses internal spatial variables.
        norder should be 0, 2 or 4.
        '''
        if(norder!=0 and norder!=2 and norder!=4):
            raise ValueError("norder={:d} not recognized; should be 0, 2 or 4.".format(norder))
        name = 'sT1{:d}'.format(norder)
        return self._bessel_array(name, _shearT1_integrand_direct, self.DT1, norder)

    def _shear_bessel_T2(self):
        ''' The quantity sT2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sT2', _shearT2_integrand, self.DT2)

    def _shear_bessel_A(self):
        ''' The quantity sA, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sA', _shearA_integrand, self.sbar)

    def _force_bessel_0(self):
        ''' The quantity f0, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('f0', _f0_integrand, self.cU)

    def _force_bessel_2(self):
        ''' The quantity f2, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('f2', _f2_integrand, self.cT1, self.cT2, self.sbar)

    def _force_bessel_3(self):
        ''' The quantity f3, defined as a Bessel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('f3', _f3_integrand, self.cT1, self.sbar)

    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _initialize_space(self):
        ''' Initialize 3D grids of b, theta and phi variables. '''
        b = np.linspace(-self.bmax, self.bmax, self.nb)
        x, y, z = np.meshgrid(b, b, b, indexing='ij')
        self.b = np.sqrt(x**2 + y**2 + z**2)
        self.theta = np.arctan2(np.sqrt(x**2+y**2 + 1e-9), z)
        self.phi = np.arctan2(y, x)
        self.x = b # a 1D array to grab for making plots
        return

    def _initialize_bessel(self):
        ''' Initialize a dict to contain cached Bessel transform arrays. '''
        self.bessel_cache = {}
        return

    def _bessel_array(self, name, integrand, *args):
        ''' A method to retrieve a particular Bessel transform array,
        or to create it if it doesn't exist.
        '''
        if(name not in self.bessel_cache):
            path = self._cache_path_bessel(name)
            if(path.is_file()):
                self.bessel_cache[name] = np.load(path)
            else:
                self.bessel_cache[name] = quad_vec(integrand, self.kmin, self.kmax,
                                                   args=(self.b, *args),
                                                   workers=8)[0]
                np.save(path, self.bessel_cache[name])
        return self.bessel_cache[name]

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
    unique = k**2/(2*mN)
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
    unique = -k**2/(8*mN**2)
    bessel = jn(2, k*b/hbar)
    form = k**2/(12*mN)*DT1(k) + 2*mN*cT1(k)
    return common * unique * bessel * form

def _shearT1_integrand_direct(k, b, DT1, norder):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = (-1)**(norder//2) * k**2/(8*mN**2)
    bessel = jn(norder, k*b/hbar)
    form = k**2/(8*mN)*DT1(k)
    return common * unique * bessel * form

# Integrands for force distributions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _f0_integrand(k, b, cU):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(1, k*b/hbar)
    form = cU(k)
    return common * unique * bessel * form

def _f2_integrand(k, b, cT1, cT2, sbar):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(1, k*b/hbar)
    form = cT2(k) - k**2/(8*mN**2)*sbar(k) - k**2/(20*mN**2)*(cT1(k)-sbar(k))
    return common * unique * bessel * form

def _f3_integrand(k, b, cT1, sbar):
    common = k**2/(2*np.pi**2*hbar**3)
    unique = 2*mN*k/hbar # aiming for GeV/fm**4
    bessel = jn(3, k*b/hbar)
    form = k**2/(8*mN**2)*(cT1(k)-sbar(k))
    return common * unique * bessel * form

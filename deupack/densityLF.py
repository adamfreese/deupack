# density.py
# Created 2026.08.11 by Alan Sosa
#
# This file computes LF densities

import numpy as np
import pandas as pd

from scipy.special import jv as jn
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline

from pathlib import Path

from .constants import mN, hbar
from .wf.chooser import choose_wf
from .emtff.nucleon.chooser import choose_nff
from . import emtff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density class

class DensityLF:
    ''' A class for the calculation of LF densities (spin 1/2 for now).

    This is implemented as a class so that lookup tables for mechanical form
    factors can be cached, and so that the user can create different objects
    with different EMT-FFs in their cache. Additionally, the density functions
    can themselves be cached on a per-object basis to reduce computation time
    when creating 2D density plots.
    '''

    def __init__(self,
                 nff='ba',
                 nk=4000,
                 nb=101,
                 bmax=2,    # fm
                 kmin=1e-6, # GeV
                 kmax=100    # GeV
                 ,Pplus=mN/np.sqrt(2.) #rest frame Pplus as default
                 ,S=(0.,0.,0.) #spin vector for azimuthalization direction default to unazimuthalized
                 ):
        self.nff  = choose_nff(nff)
        self.nk   = nk
        self.nb   = nb
        self.bmax = bmax
        self.kmin = kmin
        self.kmax = kmax
        self.P =Pplus

        # spin density matrix
        self.S = np.asarray(S, dtype=float)

        I = np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)
        sigma_x = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)

        sigma_y = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)

        sigma_z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)

        self.rho= 0.5*(I + sigma_x*self.S[0] +sigma_y*self.S[1]+sigma_z*self.S[2] )
        

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


    # 2D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

    def radial_force(self, pol='U'):
        ''' Radial force density, in GeV/fm**3. '''
        if(pol=='U'):
            return self._force_bessel_U()
        else:
            self._pol_error(pol)


    # def azimuthal_force(self, pol='U'):
    #     ''' azimuthal force density, in GeV/fm**3. '''
    #     theta_dep = -3 * np.sin(self.theta) * np.cos(self.theta)
    #     b_dep = self._force_bessel_2() - 2/5*self._force_bessel_3()
    #     force = theta_dep*b_dep
    #     if(pol=='U'):
    #         force *= 0
    #     elif(pol=='T'):
    #         pass
    #     elif(pol==0):
    #         force *= -1/3
    #     elif(pol==1 or pol==-1):
    #         force *= 1/6
    #     else:
    #         self._pol_error(pol)
    #     return force

    def isoradial_pressure(self, pol='U'):
        ''' Principal stress closest to the radial direction, in GeV/fm**2. '''
        pr = self.radial_pressure(pol=pol)
        pa = self.azimuthal_pressure(pol=pol)
        s = self.symmetric_shear(pol=pol)
        return 0.5*(pr+pa + np.sqrt((pr-pa)**2+4*s**2))

    def isoazimuthal_pressure(self, pol='U'):
        ''' Principal stress closest to the azimuthal direction, in GeV/fm**2. '''
        pr = self.radial_pressure(pol=pol)
        pa = self.azimuthal_pressure(pol=pol)
        s = self.symmetric_shear()
        return 0.5*(pr+pa - np.sqrt((pr-pa)**2+4*s**2))


    # Principal axes of the symmetric stress tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def e_plus(self, pol='U'):
        ''' Returns two 2D numpy arrays, with the Cartesian x, y 
        components of the isoradial principal axis.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.azimuthal_pressure(pol=pol)
        s  = self.symmetric_shear( pol=pol)
        sgn = np.sign(s)
        R  = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        X = (R*np.cos(self.phi) - sgn*Th*np.sin(self.phi))
        Y = (R*np.sin(self.phi) + sgn*Th*np.cos(self.phi))
        return X, Y

    def e_minus(self, pol='U'):
        ''' Returns two 2D numpy arrays, with the Cartesian x, y 
        components of the isoazimuthal principal axis.
        '''
        pr = self.radial_pressure( pol=pol)
        pt = self.azimuthal_pressure(pol=pol)
        s  = self.symmetric_shear( pol=pol)
        sgn = np.sign(s)
        R  = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        X = (R*np.cos(self.phi) + sgn*Th*np.sin(self.phi))
        Y = (R*np.sin(self.phi) - sgn*Th*np.cos(self.phi))
        return X, Y






    #2D density methods
    # 
    def radial_pressure(self, pol='U'):
        ''' Radial pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() + 1/2*self._shear_bessel_U()
        
        else:
            self._pol_error(pol)

    def azimuthal_pressure(self, pol='U'):
        ''' Lateral pressure in GeV/fm**3. '''
        if(pol=='U'):
            return self._pressure_bessel_U() - 1/2*self._shear_bessel_U()
        
        else:
            self._pol_error(pol)
 


    # Hankel transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _pressure_bessel_U(self):
        ''' The quantity pU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pU', _pressureU_integrand, self.P, self.D, self.c)

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sU', _shearU_integrand, self.P, self.D)

    def _force_bessel_U(self):
        ''' The quantity fU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('fU', _fU_integrand, self.P, self.c)
    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _initialize_space(self):
        ''' Initialize 2D grids of b and phi variables. '''
        b = np.linspace(-self.bmax, self.bmax, self.nb)
        x, y = np.meshgrid(b, b, indexing='ij')
        self.b = np.sqrt(x**2 + y**2 )
        # self.theta = np.arctan2(np.sqrt(x**2+y**2 + 1e-9), z)
        self.phi = np.arctan2(y, x)
        self.x = b # a 1D array to grab for making plots
        return
 
    def _initialize_bessel(self):
        ''' Initialize a dict to contain cached Hankel transform arrays. '''
        self.bessel_cache = {}
        return

    def _bessel_array(self, name, integrand, *args):
        ''' A method to retrieve a particular Hankel transform array,
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
        filename = "Nucleon_emtff_table_{}_{:d}_{:.2e}_{:.2e}".format(
         self.nff.name, self.nk, self.kmin, self.kmax
        )
        path = Path(__file__).parent / 'cache/{}.csv'.format(filename)
        return path

    def _cache_path_bessel(self, name):
        filename = "besselRegular_{}_table_{}_{:d}_{:.2e}".format(
                name, self.nff.name, self.nb, self.bmax,
                )
        path = Path(__file__).parent / 'cache/{}.npy'.format(filename)
        return path

    def _init_emtff_table(self, save_table=False):
        k    = np.geomspace(self.kmin, self.kmax, self.nk)

        _nff = choose_nff(self.nff)
        A   = _nff.AN(k)
        D   = _nff.DN(k)
        c   = _nff.cN(k)
        J    = _nff.JN(k)
        S    = _nff.SN(k)
        if(save_table):
            df = pd.DataFrame(data={
                'k'    : k,
                'A'   : A,
                'D'   : D,
                'J'   : J,
                'c'  : c,
                'S'  : S
                })
            path = self._cache_path()
            df.to_csv(path, index=None)
        self.A   = CubicSpline(k, A)
        self.D   = CubicSpline(k, D)
        self.c   = CubicSpline(k, c)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        return

    def _load_emtff_table(self, filename):
        df   = pd.read_csv(filename)
        k    = df['k'].to_numpy()
        A   = df['A'].to_numpy()
        D   = df['D'].to_numpy()
        c   = df['c'].to_numpy()
        J  = df['J'].to_numpy()
        S  = df['S'].to_numpy()
        self.A   = CubicSpline(k, A)
        self.D   = CubicSpline(k, D)
        self.c   = CubicSpline(k, c)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        return

    def _pol_error(self,pol):
        raise ValueError(
                "pol={} not recognized; use 'U', 'V', 0.5,or -0.5".format(pol)
                )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Integrand functions
# Need to make these separate functions to use quad_vec with workers
def _pressureU_integrand(k, b, P, D, c):
    common = k/(2*np.pi*hbar**2)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(8*P)*D(k) + mN**2/(2*P)*c(k)
    return common * unique * bessel * form

def _shearU_integrand(k, b, P, D):
    common = k/(2*np.pi*hbar**2)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(4*P)*D(k)
    return common * unique * bessel * form

def _fU_integrand(k, b, P, c):
    common = k**2/(2*np.pi*hbar**3)
    unique = -mN**2/(2*P)
    bessel = jn(1, k*b/hbar)
    form = c(k)
    return common * unique * bessel * form


class DensityLFSym:
    ''' A class for the calculation of LF densities that are azimuthally symmetric (spin 1/2 for now).

    This is implemented as a class so that lookup tables for mechanical form
    factors can be cached, and so that the user can create different objects
    with different EMT-FFs in their cache. Additionally, the density functions
    can themselves be cached on a per-object basis to reduce computation time
    when creating 2D density plots.
    '''

    def __init__(self,
                 nff='ba',
                 nk=4000,
                 nb=101,
                 bmax=2,    # fm
                 kmin=1e-6, # GeV
                 kmax=100    # GeV
                 ,Pplus=mN/np.sqrt(2.) #rest frame Pplus
                 ):
        self.nff  = choose_nff(nff)
        self.nk   = nk
        self.nb   = nb
        self.bmax = bmax
        self.kmin = kmin
        self.kmax = kmax
        self.P =Pplus
        # Internal initializations of spatial variables and Bessel caches
        self._initialize_space1()
        self._initialize_bessel()
        # attempt to find a cached lookup table on disk
        path = self._cache_path()
        if(path.is_file()):
            self._load_emtff_table(path)
        else:
            # if not found, make one
            self._init_emtff_table(save_table=True)
        return

    # 2D density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def radial_force(self, pol='U'):
        ''' Radial force density, in GeV/fm**3. '''
        if(pol=='U'):
            return self._force_bessel_U()
        else:
            self._pol_error(pol)


    # Hankel transforms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _pressure_bessel_U(self):
        ''' The quantity pU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pU', _pressureU_integrand, self.P, self.D, self.c)

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sU', _shearU_integrand, self.P, self.D)

    def _force_bessel_U(self):
        ''' The quantity fU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('fU', _fU_integrand, self.P, self.c)
    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   

    def _initialize_space1(self):
        ''' Initialize 1D grids of b variables. Useful for things that only depend on radial direction '''
        x = np.linspace(0.0, self.bmax, self.nb)
        self.b = x
        return
    def _initialize_bessel(self):
        ''' Initialize a dict to contain cached Hankel transform arrays. '''
        self.bessel_cache = {}
        return

    def _bessel_array(self, name, integrand, *args):
        ''' A method to retrieve a particular Hankel transform array,
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
        filename = "Nucleon_emtff_table_{}_{:d}_{:.2e}_{:.2e}".format(
         self.nff.name, self.nk, self.kmin, self.kmax
        )
        path = Path(__file__).parent / 'cache/{}.csv'.format(filename)
        return path

    def _cache_path_bessel(self, name):
        filename = "besselRegular1D_{}_table_{}_{:d}_{:.2e}".format(
                name, self.nff.name, self.nb, self.bmax,
                )
        path = Path(__file__).parent / 'cache/{}.npy'.format(filename)
        return path

    def _init_emtff_table(self, save_table=False):
        k    = np.geomspace(self.kmin, self.kmax, self.nk)

        _nff = choose_nff(self.nff)
        A   = _nff.AN(k)
        D   = _nff.DN(k)
        c   = _nff.cN(k)
        J    = _nff.JN(k)
        S    = _nff.SN(k)
        if(save_table):
            df = pd.DataFrame(data={
                'k'    : k,
                'A'   : A,
                'D'   : D,
                'J'   : J,
                'c'  : c,
                'S'  : S
                })
            path = self._cache_path()
            df.to_csv(path, index=None)
        self.A   = CubicSpline(k, A)
        self.D   = CubicSpline(k, D)
        self.c   = CubicSpline(k, c)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        return

    def _load_emtff_table(self, filename):
        df   = pd.read_csv(filename)
        k    = df['k'].to_numpy()
        A   = df['A'].to_numpy()
        D   = df['D'].to_numpy()
        c   = df['c'].to_numpy()
        J  = df['J'].to_numpy()
        S  = df['S'].to_numpy()
        self.A   = CubicSpline(k, A)
        self.D   = CubicSpline(k, D)
        self.c   = CubicSpline(k, c)
        self.J    = CubicSpline(k, J)
        self.S    = CubicSpline(k, S)
        return

    def _pol_error(self,pol):
        raise ValueError(
                "pol={} not recognized; use 'U', 'V', 0.5,or -0.5".format(pol)
                )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Integrand functions
# Need to make these separate functions to use quad_vec with workers
def _pressureU_integrand(k, b, P, D, c):
    common = k/(2*np.pi*hbar**2)
    unique = -1
    bessel = jn(0, k*b/hbar)
    form = k**2/(8*P)*D(k) + mN**2/(2*P)*c(k)
    return common * unique * bessel * form


def _pressureV_integrand(k, b, P, D, c):
    common = k/(2*np.pi*hbar**2)
    unique = k/(2*mN)
    bessel = jn(1, k*b/hbar)
    form = k**2/(16*P)*D(k) + mN**2/(2*P)*c(k)
    return common * unique * bessel * form

def _shearU_integrand(k, b, P, D):
    common = k/(2*np.pi*hbar**2)
    unique = -1
    bessel = jn(2, k*b/hbar)
    form = k**2/(4*P)*D(k)
    return common * unique * bessel * form


def _shearV3_integrand(k, b, P, D):
    common = k/(2*np.pi*hbar**2)
    unique = k/(2*mN)
    bessel = jn(3, k*b/hbar)
    form = k**2/(4*P)*D(k)
    return common * unique * bessel * form



def _shearV_integrand(k, b, P, D):
    common = k/(2*np.pi*hbar**2)
    unique = -k/(2*mN)
    bessel = jn(1, k*b/hbar)
    form = k**2/(16*P)*D(k)
    return common * unique * bessel * form



def _fU_integrand(k, b, P, c):
    common = k**2/(2*np.pi*hbar**3)
    unique = mN**2/(2*P)
    bessel = jn(1, k*b/hbar)
    form = c(k)
    return common * unique * bessel * form


def _f0_integrand(k, b, P, c):
    common = k**2/(2*np.pi*hbar**3)
    unique = mN*k/(8*P)
    bessel = jn(0, k*b/hbar)
    form = c(k)
    return common * unique * bessel * form

def _f2_integrand(k, b, P, c):
    common = k**2/(2*np.pi*hbar**3)
    unique = -mN*k/(4*P)
    bessel = jn(2, k*b/hbar)
    form = c(k)
    return common * unique * bessel * form
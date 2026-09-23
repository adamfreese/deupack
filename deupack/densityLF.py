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
                 kmin=1e-5, # GeV
                 kmax=100    # GeV
                 ,Pplus=mN/np.sqrt(2.) #rest frame Pplus as default
                 ,SpinV=(0.,0.,0.) #spin vector for density matrix parametrization
                 ):
        self.nff  = choose_nff(nff)
        self.nk   = nk
        self.nb   = nb
        self.bmax = bmax
        self.kmin = kmin
        self.kmax = kmax
        self.P =Pplus

        # spin vector for spin density matrix
        self.Spin = np.asarray(SpinV)

        self.I = np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)
        self.sigma_x = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)

        self.sigma_y = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)

        self.sigma_z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)

        # spin density matrix
        self.rho= 0.5*(self.I + self.sigma_x*self.Spin[0] +self.sigma_y*self.Spin[1]+self.sigma_z*self.Spin[2] )
        

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
    

    def radial_force(self):
        ''' Radial force density, in GeV/fm**3. ''' 
        U =np.trace(self.rho).real
        sX =np.trace(np.matmul(self.rho,self.sigma_x)).real
        sY =np.trace(np.matmul(self.rho,self.sigma_y)).real #tracing density matrices with pauli matrices (no sigma_z depedence in force density)
        fU = U*self._force_bessel_U()
        fV0 =self._force_bessel_V0()
        fV2= self._force_bessel_V2()
        spin_thetaDepend = sX*np.sin(self.phi)-sY*np.cos(self.phi) 
        f = fU + spin_thetaDepend*(fV0+0.5*fV2)

        return f 
        


    def azimuthal_force(self):
        ''' azimuthal force density, in GeV/fm**3. '''
        sX =np.trace(np.matmul(self.rho,self.sigma_x)).real
        sY =np.trace(np.matmul(self.rho,self.sigma_y)).real #tracing density matrices with pauli matrices (no sigma_z depedence in force density)
        fV0 =self._force_bessel_V0()
        fV2= self._force_bessel_V2()
        spin_thetaDepend = sY*np.sin(self.phi)+sX*np.cos(self.phi)
        f = spin_thetaDepend*(fV0-0.5*fV2)

        return f 
         

    def isoradial_pressure(self):
        ''' Principal stress closest to the radial direction, in GeV/fm**2. '''
        pr = self.radial_pressure()
        pa = self.azimuthal_pressure()
        s = self.symmetric_shear()
        return 0.5*(pr+pa + np.sqrt((pr-pa)**2+4*s**2))

    def isoazimuthal_pressure(self):
        ''' Principal stress closest to the azimuthal direction, in GeV/fm**2. '''
        pr = self.radial_pressure()
        pa = self.azimuthal_pressure()
        s = self.symmetric_shear()
        return 0.5*(pr+pa - np.sqrt((pr-pa)**2+4*s**2))


    # Principal axes of the symmetric stress tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def e_plus(self):
        ''' Returns two 2D numpy arrays, with the Cartesian x, y 
        components of the isoradial principal axis.
        '''
        pr = self.radial_pressure()
        pt = self.azimuthal_pressure()
        s  = self.symmetric_shear()
        sgn = np.sign(s)
        tol = 1e-7
        small_s = np.abs(s) < tol
        sgn = np.where(small_s, 1.0, np.sign(s))

        R  = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        X = (R*np.cos(self.phi) - sgn*Th*np.sin(self.phi))
        Y = (R*np.sin(self.phi) + sgn*Th*np.cos(self.phi))

        # this makes sure when spin along z that we have principal stresses along right directions
        X1 = np.where(small_s, np.cos(self.phi), X)
        Y1 = np.where(small_s, np.sin(self.phi), Y)
        return X1, Y1

    def e_minus(self):
        ''' Returns two 2D numpy arrays, with the Cartesian x, y 
        components of the isoazimuthal principal axis.
        '''
        pr = self.radial_pressure()
        pt = self.azimuthal_pressure()
        s  = self.symmetric_shear()

        sgn = np.sign(s)
        tol = 1e-7
        small_s = np.abs(s) < tol
        sgn = np.where(small_s, 1.0, np.sign(s))

        
        R  = np.sqrt( 0.5*(1 - (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        Th = np.sqrt( 0.5*(1 + (pr-pt) / np.sqrt((pr-pt)**2 + 4*s**2)) )
        X = (R*np.cos(self.phi) + sgn*Th*np.sin(self.phi))
        Y = (R*np.sin(self.phi) - sgn*Th*np.cos(self.phi))

        
        # this makes sure when spin along z that we have principal stresses along right directions
        X1 = np.where(small_s, np.sin(self.phi), X)
        Y1 = np.where(small_s, -np.cos(self.phi), Y)
        return X1, Y1






    #2D density methods
    # 
    def radial_pressure(self):
        ''' Radial pressure in GeV/fm**2. '''
        U =np.trace(self.rho).real
        sX =np.trace(np.matmul(self.rho,self.sigma_x)).real
        sY =np.trace(np.matmul(self.rho,self.sigma_y)).real #tracing density matrices with pauli matrices (no sigma_z depedence in pressure density)
        
        pV =self._pressure_bessel_V()
        sV= self._shear_bessel_V()
        sV3= self._shear_bessel_V3()
        spin_thetaDepend = sX*np.sin(self.phi)-sY*np.cos(self.phi) 
        pU =U*(self._pressure_bessel_U() + 1/2*self._shear_bessel_U())

        pVec = 0.25*(4*pV+8*sV+sV3)*spin_thetaDepend


        p=pU+pVec
        return p

    def azimuthal_pressure(self):
        ''' Lateral pressure in GeV/fm**2. '''
        U =np.trace(self.rho).real
        sX =np.trace(np.matmul(self.rho,self.sigma_x)).real
        sY =np.trace(np.matmul(self.rho,self.sigma_y)).real #tracing density matrices with pauli matrices (no sigma_z depedence in pressure density)
        
        pV =self._pressure_bessel_V()
        sV3= self._shear_bessel_V3()
        spin_thetaDepend = sX*np.sin(self.phi)-sY*np.cos(self.phi) 
        pU =U*(self._pressure_bessel_U() - 1/2*self._shear_bessel_U())

        pVec = 0.25*(4*pV-sV3)*spin_thetaDepend


        p=pU+pVec
        return p
    
    def symmetric_shear(self):
        ''' Lateral pressure in GeV/fm**2. '''
        sX =np.trace(np.matmul(self.rho,self.sigma_x)).real
        sY =np.trace(np.matmul(self.rho,self.sigma_y)).real #tracing density matrices with pauli matrices (no sigma_z depedence in pressure density)
        
        sV= self._shear_bessel_V()
        sV3= self._shear_bessel_V3()
        spin_thetaDepend = sX*np.cos(self.phi)+sY*np.sin(self.phi) 

        pVec = 0.25*(4*sV-sV3)*spin_thetaDepend


        p=pVec
        return p
 
 


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

    
    def _pressure_bessel_V(self):
        ''' The quantity pV, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('pV', _pressureV_integrand, self.P, self.D, self.c)

    def _shear_bessel_U(self):
        ''' The quantity sU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sU', _shearU_integrand, self.P, self.D)
    
    def _shear_bessel_V(self):
        ''' The quantity sV, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sV', _shearV_integrand, self.P, self.D)
    
    def _shear_bessel_V3(self):
        ''' The quantity sV3, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('sV3', _shearV3_integrand, self.P, self.D)

    def _force_bessel_U(self):
        ''' The quantity fU, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('fU', _fU_integrand, self.P, self.c)
    
    def _force_bessel_V0(self):
        ''' The quantity f_V0, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('fV0', _f0_integrand, self.P, self.c)

    
    def _force_bessel_V2(self):
        ''' The quantity f_V2, defined as a Hankel transform.
        Uses internal spatial variables.
        '''
        return self._bessel_array('fV2', _f2_integrand, self.P, self.c)

    
    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _initialize_space(self):
        ''' Initialize 2D grids of b and phi variables. '''
        b = np.linspace(-self.bmax, self.bmax, self.nb)
        x, y = np.meshgrid(b, b, indexing='ij')
        self.b = np.sqrt(x**2 + y**2 )
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
                name, self.nff.name, self.nb, self.bmax
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

    def _pol_error(self):
        raise ValueError(
                "pol={} not recognized; use for unpolarized use '(0,0,0)',and use '(S_x,S_y,S_z)' for general ensembles".format(self.Spin)
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

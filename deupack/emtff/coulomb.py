# coulomb.py
# Created 2026.05.08 by Adam Freese
# - 2026.08.13: added unpolarized EMT-FFs
#
# Contributions to EMT-FFs from a Coulomb-like interaction.
# Assumes two bound particles with **opposite** charges.
#
# In progress...
# TODO
# - Currently unpolarized EMT-FFs only
# - Working on a better numerical implementation for DU.
#   Lookup table for integrand function is stable at least.
#   Analytic solution (involving E1) is messy and unstable.

import numpy as np
from scipy.special import spherical_jn as jn, exp1
from scipy.integrate import quad_vec

from scipy.interpolate import CubicSpline

from ..constants import hbar
from .impulse import regulate_zero # maybe put in a common utils.py file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the Coulomb contributions to the form factors.
#
# More friendly user interfaces are given in deuteron.py

def DU(k, dwf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, 0, np.inf,
                        args=(k, dwf),
                        workers=8
                        )[0]
    return integral

def cU(k, dwf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cU_integrand, 0, np.inf,
                        args=(k, dwf),
                        workers=8
                        )[0]
    return integral

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs: Integrands
# Parallelization of the integration requires the integrands to be defined
# as top-level (rather than nested) functions.

def _DU_integrand(r, k, dwf):
    kfm = k/hbar
    intd_self = np.pi*dwf.mNfm*dwf.alpha/kfm * jn(0,kfm*r/2) * dwf.u(r)**2
    intd_cross = -4*dwf.mNfm*dwf.alpha/kfm * _Phi2(kfm*r/2) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf):
    kfm = k/hbar
    intd_self = 0
    intd_cross = dwf.alpha/kfm/dwf.mNfm * jn(1,kfm*r/2) / r**2 * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions within integrands. In progress.

def _Phi1(z):
    ''' An auxiliary function appearing in the Coulomb D-term integrand. '''
    # TODO: better implementation
    if(z > 100):
        return 0
    z1 = z*(1+1j)
    z2 = z*(-1+1j)
    cterm1 = exp1(z1)*np.exp(z)
    cterm2 = exp1(z2)*np.exp(-z)
    rterm = np.pi/2*np.exp(-z)
    result = rterm - np.imag(cterm1-cterm2)
    return result/z

_Phi1 = np.vectorize(_Phi1)

def _Phi2(z):
    ''' An auxiliary function appearing in the Coulomb D-term integrand. '''
    result = 1/2 * ( 3*jn(1,z)/z**2 - jn(0,z)/z + _Phi1(z) )
    #result = 1/2 * ( 3*jn(1,z)/z**2 - jn(0,z)/z + Phi1(z) )
    return result

def _zPhi0_intd(y, z):
    piece0 = (z*np.sqrt(1-y**2) + 1)*jn(0,y*z)
    piece1 = 3*z*y*jn(1,y*z)
    piece2 = -2*(z*np.sqrt(1-y**2) + 1)*jn(2,y*z)
    factor = np.exp(-z*np.sqrt(1-y**2))
    return factor*(piece0 + piece1 + piece2)

def _zPhi1_intd(y, z):
    return np.exp(-z*np.sqrt(1-y**2))*jn(0,y*z) / np.sqrt(1+1e-12-y**2)

def _zPhi2_intd(y, z):
    piece0 = z*np.sqrt(1-y**2)*jn(0,y*z)
    piece1 = 2*z*y*jn(1,y*z)
    piece2 = -(z*np.sqrt(1-y**2) + 1)*jn(2,y*z)
    factor = np.exp(-z*np.sqrt(1-y**2))
    return factor*(piece0 + piece1 + piece2)

def _create_phi_tables():
    zmax = 100
    zmin = 0
    nz = 9000
    zz = np.linspace(zmin, zmax, nz)
    #
    #
    zPhi0_table = quad_vec(_zPhi0_intd, 0, 1,
                            args=(zz,),
                            #workers=8
                            )[0]
    zPhi1_table = quad_vec(_zPhi1_intd, 0, 1,
                            args=(zz,),
                            #workers=8
                            )[0]
    zPhi2_table = quad_vec(_zPhi2_intd, 0, 1,
                            args=(zz,),
                            #workers=8
                            )[0]
    zPhi0_spline = CubicSpline(zz, zPhi0_table)
    zPhi1_spline = CubicSpline(zz, zPhi1_table)
    zPhi2_spline = CubicSpline(zz, zPhi2_table)
    return zPhi0_spline, zPhi1_spline, zPhi2_spline, zmax, zmin

_zPhi0, _zPhi1, _zPhi2, _zmax, _zmin = _create_phi_tables()

def Phi0_single(z):
    if(z > _zmax):
        return 0
    return _zPhi0(z) / z

def Phi1_single(z):
    if(z > _zmax):
        return 0
    return _zPhi1(z)

def Phi2_single(z):
    if(z > _zmax):
        return 0
    if(z == 0):
        return np.pi/4
    return _zPhi2(z) / z

#Phi0 = np.vectorize(Phi0_single)
Phi1 = np.vectorize(Phi1_single)
Phi2 = np.vectorize(Phi2_single)

def Phi0(z):
    return 3*jn(1, z)/z**2

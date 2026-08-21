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

import numpy as np
from scipy.special import spherical_jn as jn, exp1
from scipy.integrate import quad_vec

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
    intd_cross = -4*dwf.mNfm*dwf.alpha/kfm * Phi2(kfm*r/2) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf):
    kfm = k/hbar
    intd_self = 0
    intd_cross = dwf.alpha/kfm/dwf.mNfm * jn(1,kfm*r/2) / r**2 * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Auxiliary functions within integrands

def Phi1(z):
    ''' An auxiliary function appearing in the Coulomb D-term integrand.
    Defined as
        \Phi_1(z) = \int_0^1 dy e^{-z\sqrt{1-y^2}} \frac{j_0(yz)}{\sqrt{1-y^2}}
    and equal to
        \Phi_1(z) = \frac{\pi}{2z} e^{-z} - \frac{1}{z}\mathrm{Im}\left\{
        E_1(z(i+1)) e^z - E_1(z(i-1))
        \right\}
    It's not the presttiest formula, but I've failed to find a simplification.

    At z=0, Phi1(0) = pi/2. This is implemented directly.

    For 0 < z <= 700, the exact formula works fine. However, at z > 700 or so,
    np.exp(z) gives an overflow and the function gives invalid results.
    This messes up integrals containing Phi1. To avoid numerical instability,
    I use an asymptotic form of Phi1 whenever z >= 50. This is far lower then
    needed, but the asymptotic formula is already an excellent approximation
    at this point.
    '''
    _zsplit = 50
    Phi = np.zeros(z.shape)
    # Region 0 (z==0)
    Phi[z==0] = np.pi/2
    # Region 1 (z > 0 and z < _zsplit)
    z1 = z[(z > 0) & (z < _zsplit)]
    Phi[(z > 0) & (z < _zsplit)] = (
            np.pi * np.exp(-z1) / 2
            - np.imag(
                exp1(z1*(1j+1)) * np.exp(z1)
                -
                exp1(z1*(1j-1)) * np.exp(-z1)
                )
            ) / z1
    # Region 2 (z >= _zsplit)
    z2 = z[z >= _zsplit]
    Phi[np.where(z >= _zsplit)] = (
            np.pi/2*np.exp(-z2)
            + np.sin(z2)/z2
            - np.cos(z2)/z2**2
            - np.sin(z2)/(2*z2**3)
            ) / z2
    return Phi

def Phi2(z):
    ''' An auxiliary function appearing in the Coulomb D-term integrand. '''
    result = 1/2 * ( 3*jn(1,z)/z**2 - jn(0,z)/z + Phi1(z) )
    return result

# yukawa.py
# Created 2026.08.21 by Adam Freese
#
# Contributions to EMT-FFs from a Yukawa-like interaction.
#
# In progress...
# TODO
# - mass terms
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
    z = kfm*r/2
    w = 4*dwf.mu**2/k**2
    intd_self = -2*dwf.mN*dwf.alpha/k*(
            (1-w)*np.arcsin(1/np.sqrt(1+w)) + np.sqrt(w)
            ) * dwf.u(r)**2 * jn(0,z)
    intd_cross = -4*dwf.mN*dwf.alpha/k*(
            Phi2(z,w) - w*Phi1(z,w)
            ) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf):
    kfm = k/hbar
    z = kfm*r/2
    w = 4*dwf.mu**2/k**2
    intd_self = dwf.alpha*dwf.mu**2/k**2/(4*np.pi*kfm) * jn(0,z) * dwf.u(r)**2
    intd_cross = k*dwf.alpha/(12*dwf.mN) * (
            Phi0(z,w) - 3/2*w*Phi1(z,w)
            ) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Auxiliary functions within integrands

def Phi0(z, w):
    ''' An auxiliary function appearing in the Yukawa EMT-FF integrands. '''
    return 3/z * (1/z + np.sqrt(w)) * jn(1,z) * np.exp(-np.sqrt(w)*z)

def Phi1(z, w):
    ''' An auxiliary function appearing in the Yukawa EMT-FF integrands.
    Defined as
        \Phi_1(z,w) = \int_0^1 dy e^{-z\sqrt{1-y^2+w}} \frac{j_0(yz)}{\sqrt{1-y^2+w}}
    and equal to
        \Phi_1(z,w) = \frac{\pi}{2X} e^{-X}
            - \frac{1}{X}\mathrm{Im}\left\{
                E_1(z(i+\sqrt{1+w}+\sqrt{w})) e^X
                - E_1(z(i-\sqrt{1+w}+\sqrt{w})) e^{-X}
            \right\}
    where
        X = \sqrt{1 + w} z
    It's not the presttiest formula, but I've failed to find a simplification.

    At z=0, Phi1(0,w) = asin(1/sqrt(1+w)). This is implemented directly.

    Intermediate X, the exact formula works fine. For very large X,
    np.exp(X) gives an overflow and the function gives invalid results.
    This messes up integrals containing Phi1. To avoid numerical instability,
    I use an asymptotic form of Phi1 whenever z >= 50. This is far lower then
    needed, but the asymptotic formula is already an excellent approximation
    at this point.
    '''
    # TODO: double-check this
    # w = 0 result
    _zsplit = 50
    Phi = np.zeros(z.shape)
    X = np.sqrt(1+w) * z
    # Region 0 (z==0)
    Phi[z==0] = np.arcsin(1/np.sqrt(1+w[z==0]))
    # Region 1 (z > 0 and z < _zsplit)
    z1 = z[(z > 0) & (X < _zsplit)]
    w1 = w[(z > 0) & (X < _zsplit)]
    X1 = X[(z > 0) & (X < _zsplit)]
    Phi[(X > 0) & (X < _zsplit)] = (
            np.pi * np.exp(-X1) / 2
            - np.imag(
                exp1(z1*(1j+np.sqrt(1+w1)+np.sqrt(w1))) * np.exp(X1)
                -
                exp1(z1*(1j-np.sqrt(1+w1)+np.sqrt(w1))) * np.exp(-X1)
                )
            ) / X1
    # Region 2 (z >= _zsplit)
    # TODO: work out analytically
    # Lazy implementation for now ...
    z2 = z[X >= _zsplit]
    X2 = X[X >= _zsplit]
    w2 = w[X >= _zsplit]
    x2a = 1j+np.sqrt(1+w2)+np.sqrt(w2)
    x2b = 1j-np.sqrt(1+w2)+np.sqrt(w2)
    Phi[np.where(X >= _zsplit)] = (
            np.pi/2*np.exp(-X2)
            - np.imag(
                np.exp(-1j*z2) * (
                    1/x2a*(1/x2a - 2/x2a**2 + 6/x2a**3)
                    -
                    1/x2b*(1/x2b - 2/x2b**2 + 6/x2b**3)
                    )
                ) * np.exp(-np.sqrt(w2)*z2)/X2
            )
    return Phi

def Phi2(z, w):
    ''' An auxiliary function appearing in the Yukawa EMT-FF integrands. '''
    result = 1/2 * ( Phi0(z,w) - jn(0,z)/z*np.exp(-z*np.sqrt(w)) + (1+w)*Phi1(z,w) )
    return result

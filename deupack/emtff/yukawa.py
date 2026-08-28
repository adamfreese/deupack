# yukawa.py
# Created 2026.08.21 by Adam Freese
#
# Contributions to EMT-FFs from a Yukawa-like interaction.
#
# In progress...
# TODO
# - Currently unpolarized EMT-FFs only
# - Have switches for field spin, like/unlike charges
# - What if there's more than one Yukawa field?

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
    intd_cross = -2*dwf.mNfm*dwf.alpha/kfm**2*(
            kfm*(1-w)*Phi(z,w)/2
            +
            2*(
                6*(1+dwf.mu*r/hbar)*jn(1,kfm*r/2)/(kfm*r)
                -
                jn(0,kfm*r/2)
                ) * np.exp(-dwf.mu*r/hbar) / r
            ) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf):
    kfm = k/hbar
    intd = dwf.alpha/(dwf.mNfm*kfm)* dwf.u(r)**2 * (
            (1+dwf.mu*r/hbar)*np.exp(-dwf.mu*r/hbar)/r**2 * jn(1,kfm*r/2)
            )
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Auxiliary function

def Phi(z, w):
    ''' An auxiliary function appearing in the Yukawa interference D-term.
    Defined as
        \Phi_1(z,w) = \int_{-1}^1 dy e^{-z\sqrt{1-y^2+w}} \frac{j_0(yz)}{\sqrt{1-y^2+w}}
    and equal to
        \Phi_1(z,w) = \frac{\pi}{X} e^{-X}
            - \frac{1}{X}\mathrm{Im}\left\{
                E_1(z(i+\sqrt{1+w}+\sqrt{w})) e^X
                - E_1(z(i-\sqrt{1+w}+\sqrt{w})) e^{-X}
            \right\}
    where
        X = \sqrt{1 + w} z
    It's not the presttiest formula, but I've failed to find a simplification.

    At z=0, Phi(0,w) = 2*asin(1/sqrt(1+w)). This is implemented directly.

    Intermediate X, the exact formula works fine. For very large X,
    np.exp(X) gives an overflow and the function gives invalid results.
    This messes up integrals containing Phi. To avoid numerical instability,
    I use an asymptotic form of Phi whenever X >= 50. This is far lower then
    needed, but the asymptotic formula is already an excellent approximation
    at this point.
    '''
    # w = 0 result
    _zsplit = 50
    Phi = np.zeros(z.shape)
    X = np.sqrt(1+w) * z
    # Region 0 (z==0)
    Phi[z==0] = 2*np.arcsin(1/np.sqrt(1+w[z==0]))
    # Region 1 (z > 0 and z < _zsplit)
    z1 = z[(z > 0) & (X < _zsplit)]
    w1 = w[(z > 0) & (X < _zsplit)]
    X1 = X[(z > 0) & (X < _zsplit)]
    Phi[(X > 0) & (X < _zsplit)] = (
            np.pi * np.exp(-X1)
            - 2*np.imag(
                exp1(z1*(1j+np.sqrt(1+w1)+np.sqrt(w1))) * np.exp(X1)
                -
                exp1(z1*(1j-np.sqrt(1+w1)+np.sqrt(w1))) * np.exp(-X1)
                )
            ) / X1
    # Region 2 (z >= _zsplit)
    # Lazy implementation
    z2 = z[X >= _zsplit]
    X2 = X[X >= _zsplit]
    w2 = w[X >= _zsplit]
    x2a = 1j+np.sqrt(1+w2)+np.sqrt(w2)
    x2b = 1j-np.sqrt(1+w2)+np.sqrt(w2)
    Phi[np.where(X >= _zsplit)] = (
            np.pi*np.exp(-X2)
            - 2*np.imag(
                np.exp(-1j*z2) * (
                    1/x2a*(1/x2a - 2/x2a**2 + 6/x2a**3)
                    -
                    1/x2b*(1/x2b - 2/x2b**2 + 6/x2b**3)
                    )
                ) * np.exp(-np.sqrt(w2)*z2)/X2
            )
    return Phi

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
            kfm*(1-w)*Phi(z,w,0)/2
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

def Phi(zeta, omega, delta):
    r"""
    An auxiliary function appearing in the Yukawa interference D-term.
    Defined as

    .. math::
        \Phi_1(\zeta,\omega,\delta)
        =
        \int_{-1}^1 dy
        \frac{ e^{-\zeta\sqrt{1-y^2+\omega}} }{\sqrt{1-y^2+\omega}}
        j_0\big([y+\delta]\zeta\big)

    and equal to

    .. math::
        \Phi(\zeta,\omega,\delta)
        =
        \frac{1}{\zeta\sqrt{1+\omega-\delta^2}}
        \left\{
          \e^{-\zeta\sqrt{1+\omega-\delta^2}}
          \Big( \mathrm{Im}\big[ E_1(Z_1) - E_1(Z_2) \big] + \pi \Big)
          +
          \e^{+\zeta\sqrt{1+\omega-\delta^2}}
          \mathrm{Im}\big[ E_1(Z_3) - E_1(Z_4) \big]
          \right\}

    where

    .. math::
        Z_1 = \zeta[\sqrt{\omega} - \sqrt{1+\omega-\delta^2} + i(1+\delta)]

        Z_2 = \zeta[\sqrt{\omega} - \sqrt{1+\omega-\delta^2} - i(1-\delta)]

        Z_3 = \zeta[\sqrt{\omega} + \sqrt{1+\omega-\delta^2} - i(1-\delta)]

        Z_4 = \zeta[\sqrt{\omega} + \sqrt{1+\omega-\delta^2} + i(1+\delta)]

    It's not the presttiest formula, but I've failed to find a simplification.

    At z=0, Phi(0,omega,delta) = 2*asin(1/sqrt(1+omega)). This is implemented directly.

    For intermediate X=zeta*sqrt(1+omega-delta**2), the exact formula works fine.
    For very large X, np.exp(X) overflows. This messes up integrals containing Phi.
    To avoid numerical instability, I use an asymptotic form whenever X>=50.
    """
    # w = 0 result
    _split = 50
    Phi = np.zeros(zeta.shape)
    X = np.sqrt(1+omega-delta**2) * zeta
    # Region 0 (zeta==0)
    Phi[zeta==0] = 2*np.arcsin(1/np.sqrt(1+omega[zeta==0]))
    # Region 1 (zeta > 0 and X < _split)
    zeta_anl  = zeta[ (zeta > 0) & (X < _split)]
    omega_anl = omega[(zeta > 0) & (X < _split)]
    Phi[(X > 0) & (X < _split)] = Phi_analytic(zeta_anl, omega_anl, 0)
    # Region 2 (z >= _split)
    zeta_asy  = zeta[ (zeta > 0) & (X >= _split)]
    omega_asy = omega[(zeta > 0) & (X >= _split)]
    Phi[np.where(X >= _split)] = Phi_asymptotic(zeta_asy, omega_asy, 0)
    return Phi

def Phi_analytic(zeta, omega, delta):
    ''' Exact analytic result for Phi function.
    See Phi docstring for more details.
    '''
    s = np.sqrt(omega)
    c = np.sqrt(1+omega-delta**2)
    Z1 = zeta*(s-c+1j*(1+delta))
    Z2 = zeta*(s-c-1j*(1-delta))
    Z3 = zeta*(s+c-1j*(1-delta))
    Z4 = zeta*(s+c+1j*(1+delta))
    return (
            np.pi * np.exp(-zeta*c)
            + np.imag(
                exp1(Z1) * np.exp(-zeta*c)
                -
                exp1(Z2) * np.exp(-zeta*c)
                +
                exp1(Z3) * np.exp(zeta*c)
                -
                exp1(Z4) * np.exp(zeta*c)
                )
            ) / (zeta*c)

def Phi_asymptotic(zeta, omega, delta):
    ''' Asymptotic form of Phi function when
        zeta*sqrt(1+omega-delta**2)
    is large. See Phi docstring for more details.
    '''
    s = np.sqrt(omega)
    c = np.sqrt(1+omega-delta**2)
    Z1 = zeta*(s-c+1j*(1+delta))
    Z2 = zeta*(s-c-1j*(1-delta))
    Z3 = zeta*(s+c-1j*(1-delta))
    Z4 = zeta*(s+c+1j*(1+delta))
    Z1_nc = zeta*(s+1j*(1+delta))
    Z2_nc = zeta*(s-1j*(1-delta))
    Z3_nc = zeta*(s-1j*(1-delta))
    Z4_nc = zeta*(s+1j*(1+delta))
    term1 = np.exp(-Z1_nc)/Z1*(1 - 1/Z1 + 2/Z1**2 - 6/Z1**3)
    term2 = np.exp(-Z2_nc)/Z2*(1 - 1/Z2 + 2/Z2**2 - 6/Z2**3)
    term3 = np.exp(-Z3_nc)/Z3*(1 - 1/Z3 + 2/Z3**2 - 6/Z3**3)
    term4 = np.exp(-Z4_nc)/Z4*(1 - 1/Z4 + 2/Z4**2 - 6/Z4**3)
    return (
            np.pi * np.exp(-zeta*c)
            + np.imag( term1 - term2 + term3 - term4)
            ) / (zeta*c)

def Phi_numeric(zeta, omega, delta):
    ''' Evaluates the Phi function by doing the numerical integral.
    See Phi docstring for more details.
    '''
    _eps = 1e-9
    integral = quad_vec(_Phi_integrand, -1+_eps, 1-_eps,
                        args=(zeta, omega, delta),
                        workers=8
                        )[0]
    return integral

def _Phi_integrand(y, zeta, omega, delta):
    c = np.sqrt(1-y**2+omega)
    return np.exp(-zeta*c) *jn(0, zeta*(y+delta)) / c

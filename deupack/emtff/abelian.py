# abelian.py
# Created 2026.08.21 by Adam Freese
#
# Contributions to EMT-FFs from Abelian static fields
#
# originally created just for spin-zero Yukawa forces for like charges
# renamed from yukawa.py to abelian.py on 2026.09.02
# as of 2026.09.02, unifies all Abelian fields, making coulomb.py obsolete
#
# TODO:
# - Polarized EMT-FFs (DT1, DT2, cT1 and cT2)

import numpy as np
from scipy.special import spherical_jn as jn, exp1
from scipy.integrate import quad_vec

from ..constants import hbar, alphaQED
from .impulse import regulate_zero # maybe put in a common utils.py file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Default values for the interaction parameters

# Assume electrostatic attraction by default
_g1_default = -np.sqrt(4*np.pi*alphaQED)
_g2_default =  np.sqrt(4*np.pi*alphaQED)
_mf_default = 0
_s_default  = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the Abelian field contributions to the form factors
#
# More friendly user interfaces are given in deuteron.py

def DU(k, dwf, field):
    k = regulate_zero(k) # avoid division by zero
    # Get field interaction parameters
    g1 = field.get('g1', _g1_default)
    g2 = field.get('g2', _g2_default)
    mf = field.get('mf', _mf_default)
    s  = field.get('s',  _s_default)
    integral = quad_vec(_DU_integrand, 0, np.inf,
                        args=(k, dwf, g1, g2, mf),
                        workers=8
                        )[0]
    return integral * (-1)**s

def cU(k, dwf, field):
    k = regulate_zero(k) # avoid division by zero
    g1 = field.get('g1', _g1_default)
    g2 = field.get('g2', _g2_default)
    mf = field.get('mf', _mf_default)
    s  = field.get('s',  _s_default)
    integral = quad_vec(_cU_integrand, 0, np.inf,
                        args=(k, dwf, g1, g2, mf),
                        workers=8
                        )[0]
    return integral * (-1)**s

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs: Integrands
# Parallelization of the integration requires the integrands to be defined
# as top-level (rather than nested) functions.

def _DU_integrand(r, k, dwf, g1, g2, mf):
    kfm = k/hbar
    z = kfm*r/2
    w = 4*mf**2/k**2
    intd_self = -dwf.mN*(g1**2 + g2**2)/(4*np.pi*k)*(
            (1-w)*np.arctan2(0.5*k,mf) + np.sqrt(w)
            ) * dwf.u(r)**2 * jn(0,z)
    intd_cross = -2*dwf.mNfm*g1*g2/(4*np.pi*kfm**2)*(
            kfm*(1-w)*Phi(z,w,0)/2
            +
            2*(
                6*(1+mf*r/hbar)*jn(1,kfm*r/2)/(kfm*r)
                -
                jn(0,kfm*r/2)
                ) * np.exp(-mf*r/hbar) / r
            ) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf, g1, g2, mf):
    kfm = k/hbar
    intd = g1*g2/(4*np.pi*dwf.mNfm*kfm)* dwf.u(r)**2 * (
            (1+mf*r/hbar)*np.exp(-mf*r/hbar)/r**2 * jn(1,kfm*r/2)
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
    # TODO: large omega and small zeta at the same time is still unstable
    # TODO: less stupid way to deal with scalar argumetns
    if(np.isscalar(zeta)):
        zeta = np.array([zeta])
    if(np.isscalar(omega)):
        omega = np.array([omega])
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
    ZB = zeta*(s-c+1j*(1+delta))
    ZA = zeta*(s-c-1j*(1-delta))
    ZC = zeta*(s+c-1j*(1-delta))
    ZD = zeta*(s+c+1j*(1+delta))
    return (
            np.pi * np.exp(-zeta*c)
            + np.imag(
                exp1(ZB) * np.exp(-zeta*c)
                -
                exp1(ZA) * np.exp(-zeta*c)
                +
                exp1(ZC) * np.exp(zeta*c)
                -
                exp1(ZD) * np.exp(zeta*c)
                )
            ) / (zeta*c)

def Phi_asymptotic(zeta, omega, delta):
    ''' Asymptotic form of Phi function when
        zeta*sqrt(1+omega-delta**2)
    is large. See Phi docstring for more details.
    '''
    s = np.sqrt(omega)
    c = np.sqrt(1+omega-delta**2)
    ZB = zeta*(s-c+1j*(1+delta))
    ZA = zeta*(s-c-1j*(1-delta))
    ZC = zeta*(s+c-1j*(1-delta))
    ZD = zeta*(s+c+1j*(1+delta))
    ZB_nc = zeta*(s+1j*(1+delta))
    ZA_nc = zeta*(s-1j*(1-delta))
    ZC_nc = zeta*(s-1j*(1-delta))
    ZD_nc = zeta*(s+1j*(1+delta))
    termB = np.exp(-ZB_nc)/ZB*(1 - 1/ZB + 2/ZB**2 - 6/ZB**3)
    termA = np.exp(-ZA_nc)/ZA*(1 - 1/ZA + 2/ZA**2 - 6/ZA**3)
    termC = np.exp(-ZC_nc)/ZC*(1 - 1/ZC + 2/ZC**2 - 6/ZC**3)
    termD = np.exp(-ZD_nc)/ZD*(1 - 1/ZD + 2/ZD**2 - 6/ZD**3)
    return (
            np.pi * np.exp(-zeta*c)
            + np.imag( termB - termA + termC - termD)
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

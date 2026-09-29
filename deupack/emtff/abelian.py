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
_mu_default = 0
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
    mu = field.get('mu', _mu_default)
    s  = field.get('s',  _s_default)
    # Allow user to provide a breakpoint k0 below which the small-k form shoud
    # be used. Where the numerics of the exact form break down seems to vary
    # a lot, and I can only figure it out by trial-and-error on a case-by-case
    # basis. So It's best to make it a user parameter. But if the user doesn't
    # provide a breakpoint, set k0=0 and use the exact form everywhere.
    k0 = field.get('k0', 0)
    # Break into small- and large-k regions to deal with instability in former
    if(np.isscalar(k)):
        k = np.array([k])
    k_smol = k[k < k0]
    k_beeg = k[k >= k0]
    D = np.zeros(k.shape)
    if(mu==0):
        D[k < k0] = _DU_massless_smolk(k_smol, dwf, g1, g2, s)
    else:
        D[k < k0] = _DU_massive_zero(dwf, g1, g2, mu, s)
    D[k >= k0] = _DU_exact(k_beeg, dwf, g1, g2, mu, s)
    return D

def cU(k, dwf, field):
    k = regulate_zero(k) # avoid division by zero
    g1 = field.get('g1', _g1_default)
    g2 = field.get('g2', _g2_default)
    mu = field.get('mu', _mu_default)
    s  = field.get('s',  _s_default)
    integral = quad_vec(_cU_integrand, 0, np.inf,
                        args=(k, dwf, g1, g2, mu),
                        workers=8
                        )[0]
    return integral * (-1)**s

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood details for the EMT-FFs: special cases and foward limits

def _DU_exact(k, dwf, g1, g2, mu, s):
    integral = quad_vec(_DU_integrand, 0, np.inf,
                        args=(k, dwf, g1, g2, mu),
                        workers=8
                        )[0]
    return integral * (-1)**s

def _DU_massless_smolk(k, dwf, g1, g2, s):
    common = (-1)**(s+1) * 2*dwf.mN
    coef = (g1+g2)**2/16
    integral = quad_vec(_DU0_integrand_massless, 0, np.inf,
                        args=(dwf,),
                        workers=8
                        )[0]
    finite = -g1*g2*7/(60*np.pi) * integral / hbar
    return common*(coef/k + finite)

def _DU_massive_zero(dwf, g1, g2, mu, s):
    common = (-1)**(s+1) * 2*dwf.mN / (4*np.pi)
    coef = (g1+g2)**2 / 3
    integral = quad_vec(_DU0_integrand_massive, 0, np.inf,
                        args=(dwf,mu),
                        workers=8
                        )[0]
    finite = g1*g2*integral / hbar
    return common*(coef/mu + finite)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs: integrands
# Parallelization of the integration requires the integrands to be defined
# as top-level (rather than nested) functions.

def _DU_integrand(r, k, dwf, g1, g2, mu):
    kfm = k/hbar
    z = kfm*r/2
    w = 4*mu**2/k**2
    intd_self = -dwf.mN*(g1**2 + g2**2)/(4*np.pi*k)*(
            (1-w)*np.arctan2(0.5*k,mu) + np.sqrt(w)
            ) * dwf.u(r)**2 * jn(0,z)
    intd_cross = -2*dwf.mNfm*g1*g2/(4*np.pi*kfm**2)*(
            kfm*(1-w)*Phi(z,w,0)/2
            +
            2*(
                6*(1+mu*r/hbar)*jn(1,kfm*r/2)/(kfm*r)
                -
                jn(0,kfm*r/2)
                ) * np.exp(-mu*r/hbar) / r
            ) * dwf.u(r)**2
    intd = intd_self + intd_cross
    return intd

def _cU_integrand(r, k, dwf, g1, g2, mu):
    kfm = k/hbar
    intd = g1*g2/(4*np.pi*dwf.mNfm*kfm)* dwf.u(r)**2 * (
            (1+mu*r/hbar)*np.exp(-mu*r/hbar)/r**2 * jn(1,kfm*r/2)
            )
    return intd

# Forward limit integrands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _DU0_integrand_massless(r, dwf):
    return r * dwf.u(r)**2

def _DU0_integrand_massive(r, dwf, mu):
    term1 = r/5 * np.exp(-mu*r/hbar) * (1 - mu*r/hbar/9)
    term2 = -2/(3*mu/hbar) * (1 - np.exp(-mu*r/hbar))
    return dwf.u(r)**2 * (term1 + term2)

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

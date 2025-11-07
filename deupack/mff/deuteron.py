# deuteron.py
# Created 2025.09.30 by Adam Freese
#
# This file contains formulas for form factors as given in the work by
# me, Alan Sosa, and Wim Cosyn.

import numpy as np

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec

from ..constants import mN, hbar

# Default nucleon form factors: Hackett, Pefkou & Shanahan (HPS)
from .nucleonhps import AN as _AN, JN as _JN, DN as _DN

# For the form factors not in HPS, use some simple guesses
from .nucleon import SN as _SN, cN as _cN


# import wavefunction modules (keep existing default function aliases below)
from ..wf import av18 as av18_mod
from ..wf import paris as paris_mod

# Default deuteron wave function: AV18 (keep existing default aliases for backward compat)
from ..wf.av18 import u as _u, u1 as _u1, u2 as _u2, u3 as _u3
from ..wf.av18 import w as _w, w1 as _w1, w2 as _w2, w3 as _w3

# Add mapping for convenient selection
WAVEFUNCTIONS = {
    'av18': (av18_mod.u, av18_mod.w, av18_mod.u1, av18_mod.w1,
             av18_mod.u2, av18_mod.w2, av18_mod.u3, av18_mod.w3),
    'paris': (paris_mod.u, paris_mod.w, paris_mod.u1, paris_mod.w1,
              paris_mod.u2, paris_mod.w2, paris_mod.u3, paris_mod.w3),
}

def _choose_wf(wf, u, w, u1, w1, u2, w2, u3, w3):
    """Return a tuple (u,w,u1,w1,u2,w2,u3,w3) according to wf.
    wf can be None (use provided u/w...), a string 'av18'/'paris', or a module-like object.
    """
    if wf is None:
        return (u, w, u1, w1, u2, w2, u3, w3)
    if isinstance(wf, str):
        key = wf.lower()
        if key in WAVEFUNCTIONS:
            return WAVEFUNCTIONS[key]
        raise ValueError(f"Unknown wf '{wf}', valid: {list(WAVEFUNCTIONS.keys())}")
    # module-like object with attributes u,w,u1,...
    for attr in ('u','w','u1','w1','u2','w2','u3','w3'):
        if not hasattr(wf, attr):
            raise ValueError("wf module must have attributes: u,w,u1,w1,u2,w2,u3,w3")
    return (wf.u, wf.w, wf.u1, wf.w1, wf.u2, wf.w2, wf.u3, wf.w3)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the form factors
# Add optional wf parameter (last arg) that overrides u/w function arguments if provided.

def AU(k, u=_u, w=_w, AN=_AN, wf=None):
    ''' The mechanical form factor AU.
    Pass wf='av18' or wf='paris' (or wf=av18_mod) to select wavefunction.
    '''
    u, w, *_ = _choose_wf(wf, u, w, _u1, _w1, _u2, _w2, _u3, _w3)
    return _AU(k, u=u, w=w, AN=AN)

def AT(k, u=_u, w=_w, AN=_AN, wf=None):
    u, w, *_ = _choose_wf(wf, u, w, _u1, _w1, _u2, _w2, _u3, _w3)
    return _AT(k, u=u, w=w, AN=AN)

def DU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN, DN=_DN, wf=None):
    u, w, u1, w1, u2, w2, _, _ = _choose_wf(wf, u, w, u1, w1, u2, w2, _u3, _w3)
    return _DU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN, DN=_DN, wf=None):
    u, w, u1, w1, u2, w2, _, _ = _choose_wf(wf, u, w, u1, w1, u2, w2, _u3, _w3)
    return _DT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN, wf=None):
    u, w, u1, w1, u2, w2, _, _ = _choose_wf(wf, u, w, u1, w1, u2, w2, _u3, _w3)
    return _DT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN)

def cU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN, wf=None):
    u, w, u1, w1, u2, w2, u3, w3 = _choose_wf(wf, u, w, u1, w1, u2, w2, u3, w3)
    return _cU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, cN=cN)

def cT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN, wf=None):
    u, w, u1, w1, u2, w2, u3, w3 = _choose_wf(wf, u, w, u1, w1, u2, w2, u3, w3)
    return _cT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, cN=cN)

def cT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, wf=None):
    u, w, u1, w1, u2, w2, u3, w3 = _choose_wf(wf, u, w, u1, w1, u2, w2, u3, w3)
    return _cT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN)

def J(k, u=_u, w=_w, AN=_AN, JN=_JN, wf=None):
    u, w, *_ = _choose_wf(wf, u, w, _u1, _w1, _u2, _w2, _u3, _w3)
    return _J(k, u=u, w=w, AN=AN, JN=JN)

def S(k, u=_u, w=_w, SN=_SN, wf=None):
    u, w, *_ = _choose_wf(wf, u, w, _u1, _w1, _u2, _w2, _u3, _w3)
    return _S(k, u=u, w=w, SN=SN)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the MFFs:
# 1. Integrands
#    Parallelization of the integration requires the integrands to be defined
#    as top-level (rather than nested) functions.

def _AU_integrand(r, k, u, w, AN):
    kfm = k/hbar
    intd = AN(k) * (u(r)**2 + w(r)**2)/2 * jn(0, kfm*r/2)
    return intd

def _AT_integrand(r, k, u, w, AN):
    kfm = k/hbar
    intd = AN(k) * jn(2, kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            ) * 6*mN**2/k**2
    return intd

def _DU_integrand(r, k, u, w, u1, w1, u2, w2, AN, JN, DN):
    kfm = k/hbar
    A_piece = 4*AN(k)/kfm**2*jn(2,kfm*r/2)*(
            (2*u(r)**2+8*w(r)**2)/r**2
            - (u(r)*u1(r)+w(r)*w1(r))/r
            + u(r)*u2(r) + w(r)*w2(r)
            - u1(r)**2 - w1(r)**2
            )
    J_piece = -12*JN(k)/kfm*jn(1,kfm*r/2)*w(r)**2/r
    D_piece = 2*DN(k)*jn(0,kfm*r/2)*(u(r)**2 + w(r)**2)
    intd = A_piece + J_piece + D_piece
    return intd

def _DT1_integrand(r, k, u, w, u1, w1, u2, w2, AN, JN, DN):
    kfm = k/hbar
    A_piece = 48 * (mN/k)**2 / kfm**2 *AN(k) * jn(4,kfm*r/2)*(
            np.sqrt(2)*(u(r)*w2(r) + w(r)*u2(r) - 2*u1(r)*w1(r))
            - w(r)*w2(r) + w1(r)**2
            + ( np.sqrt(2)*(3*w(r)*u1(r)-5*u(r)*w1(r)) + w(r)*w1(r) )/r
            + 6*( 2*np.sqrt(2)*u(r)*w(r) - w(r)**2 )/r**2
            )
    J_piece = 96*(mN/k)**2/kfm * JN(k) * jn(3,kfm*r/2) * (
            np.sqrt(2)*(u(r)*w1(r)-w(r)*u1(r)) - (2*np.sqrt(2)*u(r)*w(r)-w(r)**2)/r
            )
    D_piece = 24*mN**2/k**2 * DN(k) * jn(2,kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            )
    intd = A_piece + J_piece + D_piece
    return intd

def _DT2_integrand(r, k, u, w, u1, w1, u2, w2, AN, JN):
    kfm = k/hbar
    A_piece = 24/kfm**3 * AN(k) * jn(3,kfm*r/2)*(
            ( (2*np.sqrt(2)*u2(r)-w2(r))*w(r) - (2*np.sqrt(2)*u1(r)-w1(r))*w1(r)) / r
            + ( 2*np.sqrt(2)*u(r)+5*w(r) )*w1(r)/r**2
            - 18*w(r)**2/r**3
            )
    J_piece = 12/kfm**2 * JN(k) * jn(2,kfm*r/2)*(
            np.sqrt(2)*(u(r)*w2(r) - w(r)*u2(r))
            - 4*(np.sqrt(2)*u1(r) + w1(r))*w(r)/r
            -(2*np.sqrt(2)*u(r)*w(r) - w(r)**2)/r**2
            + 3*w(r)**2/r**2
            )
    intd = A_piece + J_piece
    return intd

def _cU_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    kfm = k/hbar
    A1_piece = AN(k)/(2*kfm*(mN/hbar)**2) * jn(1,kfm*r/2)*(
            2*u1(r)*u2(r) + 2*w1(r)*w2(r)
            - 12*w(r)**2/r**3
            )
    A02_piece = AN(k)/(12*(mN/hbar)**2)*(jn(0,kfm*r/2)-2*jn(2,kfm*r/2))*(
            u(r)*u2(r) + w(r)*w2(r)
            )
    c_piece = 0.5*cN(k)*jn(0,kfm*r/2)*(u(r)**2 + w(r)**2)
    intd = A1_piece + A02_piece + c_piece
    return intd

def _cT1_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    kfm = k/hbar
    A3_piece = 6*AN(k)/kfm**3 * jn(3,kfm*r/2)*(
            np.sqrt(2)*(2*u1(r)*w2(r) + 2*w1(r)*u2(r))
            - 2*w1(r)*w2(r)
            + 2*np.sqrt(2)*( u(r)*w2(r) - w(r)*u2(r) )/r
            - 6*np.sqrt(2)*( u(r)*w1(r) - w(r)*u1(r) )/r**2
            - 12*( 2*np.sqrt(2)*u(r)*w(r) - w(r)**2 )/r**3
            )
    A24_piece = 6*AN(k)/(14*kfm**2)*(3*jn(2,kfm*r/2)-4*jn(4,kfm*r/2))*(
                np.sqrt(2)*(u(r)*w2(r) + w(r)*u2(r))
                - w(r)*w2(r)
                )
    c_piece = 6*mN**2/k**2 * cN(k) * jn(2,kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            )
    intd = A3_piece + A24_piece + c_piece
    return intd

def _cT2_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN):
    kfm = k/hbar
    A2_term = jn(2,kfm*r/2)*(
            2*(w1(r)*w2(r) - np.sqrt(2)*(w1(r)*u2(r) + u1(r)*w2(r))) / r
            + (2*np.sqrt(2)*u(r) - w(r))*w2(r) / r**2
            + 12*np.sqrt(2)*w(r)*u1(r) / r**3
            - 12*( np.sqrt(2)*u(r)*w(r) + w(r)**2 ) / r**4
            )
    A13_term = kfm*(2*jn(1,kfm*r/2) - 3*jn(3,kfm*r/2))/10*(
            w2(r) - 2*np.sqrt(2)*u2(r)
            ) * w(r) / r
    intd = 3*AN(k)/(mN**2*k**2)*hbar**4 * (A2_term + A13_term)
    return intd

def _J_integrand(r, k, u, w, AN, JN):
    kfm = k/hbar
    A_piece = 9/2*AN(k)/kfm * jn(1,kfm*r/2) * w(r)**2/r
    J0_piece = JN(k)*jn(0,kfm*r/2)*(u(r)**2 - w(r)**2/2)
    J2_piece = JN(k)*jn(2,kfm*r/2)*(w(r)**2 + np.sqrt(2)*u(r)*w(r))/2
    intd = A_piece + J0_piece + J2_piece
    return intd

def _S_integrand(r, k, u, w, SN):
    kfm = k/hbar
    S0_piece = SN(k)*jn(0,kfm*r/2)*(u(r)**2 - w(r)**2/2)
    S2_piece = SN(k)*jn(2,kfm*r/2)*(w(r)**2 + np.sqrt(2)*u(r)*w(r))/2
    intd = S0_piece + S2_piece
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the MFFs:
# 2. Integration
#    quad_vec achieves good speed for parallel calculation of form factors
#    at multiple k values. It's also parallelizable.

def _AU(k, u, w, AN):
    integral = quad_vec(_AU_integrand, 0.05, np.inf,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _AT(k, u, w, AN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, 0.05, np.inf,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _DU(k, u, w, u1, w1, u2, w2, AN, JN, DN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8)[0]
    return integral * 2

def _DT1(k, u, w, u1, w1, u2, w2, AN, JN, DN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8)[0]
    return integral * 2

def _DT2(k, u, w, u1, w1, u2, w2, AN, JN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN),
                        workers=8)[0]
    return integral * 2

def _cU(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cU_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8)[0]
    return integral * 2

def _cT1(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cT1_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8)[0]
    return integral * 2

def _cT2(k, u, w, u1, w1, u2, w2, u3, w3, AN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cT2_integrand, 0.05, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN),
                        workers=8)[0]
    return integral * 2

def _J(k, u, w, AN, JN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_J_integrand, 0.05, np.inf,
                        args=(k, u, w, AN, JN),
                        workers=8)[0]
    return integral * 2

def _S(k, u, w, SN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, 0.05, np.inf,
                        args=(k, u, w, SN),
                        workers=8)[0]
    return integral * 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Misc utilities

def regulate_zero(X):
    ''' Takes a scalar or an array, and if it is or contains 0,
    the 0 is shifted.
    '''
    epsilon = 1e-6
    if(np.isscalar(X)):
        if(X==0):
            X += epsilon
    else:
        X[X==0] = epsilon
    return X

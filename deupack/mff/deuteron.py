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

# Default deuteron wave function: AV18
from ..wf.av18 import u as _u, u1 as _u1, u2 as _u2, u3 as _u3
from ..wf.av18 import w as _w, w1 as _w1, w2 as _w2, w3 as _w3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the form factors

def AU(k, u=_u, w=_w, AN=_AN):
    ''' The mechanical form factor AU.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _AU(k, u=u, w=w, AN=AN)

def AT(k, u=_u, w=_w, AN=_AN):
    ''' The mechanical form factor AT.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _AT(k, u=_u, w=_w, AN=_AN)

def DU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN, DN=_DN):
    ''' The mechanical form factor DU.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _DU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN, DN=_DN)

def DT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, DN=_DN):
    ''' The mechanical form factor DT1.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _DT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, DN=_DN)

def DT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN):
    ''' The mechanical form factor DT2.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _DT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, AN=_AN, JN=_JN)

def cU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN):
    ''' The mechanical form factor cbarU.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _cU(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN)

def cT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN):
    ''' The mechanical form factor cbarT1.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _cT1(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN, cN=_cN)

def cT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN):
    ''' The mechanical form factor cbarT2.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _cT2(k, u=_u, w=_w, u1=_u1, w1=_w1, u2=_u2, w2=_w2, u3=_u3, w3=_w3, AN=_AN)

def J(k, u=_u, w=_w, AN=_AN, JN=_JN):
    ''' The mechanical form factor J.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _J(k, u=_u, w=_w, AN=_AN, JN=_JN)

def S(k, u=_u, w=_w, SN=_SN):
    ''' The mechanical form factor S.
    Assumes k is in GeV.
    By default uses the AV18 wave function.
    '''
    return _S(k, u=_u, w=_w, SN=_SN)

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
            ) * 3*mN**2/k**2
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

def _DT1_integrand(r, k, u, w, u1, w1, u2, w2, AN, DN):
    kfm = k/hbar
    A_piece = 24 * (mN/k)**2 / kfm**2 *AN(k) * jn(4,kfm*r/2)*(
            np.sqrt(2)*(u(r)*w2(r) + w(r)*u2(r) - 2*u1(r)*w1(r))
            - w(r)*w2(r) + w1(r)**2
            + ( np.sqrt(2)*(3*w(r)*u1(r)-5*u(r)*w1(r)) + w(r)*w1(r) )/r
            + 6*( np.sqrt(2)*u(r)*w(r) - w(r)**2 )/r**2
            )
    D_piece = 12*mN**2/k**2 * DN(k) * jn(2,kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            )
    intd = A_piece + D_piece
    return intd

def _DT2_integrand(r, k, u, w, u1, w1, u2, w2, AN, JN):
    kfm = k/hbar
    A_piece = 24/kfm**3 *AN(k) * jn(3,kfm*r/2)*(
            (
                (2*np.sqrt(2)*u2(r)-w2(r))*w(r)
                - (2*np.sqrt(2)*u1(r)-w1(r))*w1(r)
                ) / r
            + ( 2*np.sqrt(2)*u(r)+5*w(r) )*w1(r)/r**2
            - 18*w(r)**2/r**3
            )
    J_piece = 36/(5*kfm) * JN(k) * jn(1,kfm*r/2) * w(r)**2/r
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
    A_piece = 3*AN(k)/kfm**3 * jn(3,kfm*r/2)*(
            np.sqrt(2)*(
                u1(r)*w2(r) + w1(r)*u2(r)
                - u(r)*w3(r) - w(r)*u3(r)
                )
            + w(r)*w3(r) - w1(r)*w2(r)
            + 2*np.sqrt(2)*( u(r)*w2(r) - w(r)*u2(r) )/r
            - 6*np.sqrt(2)*( u(r)*w1(r) - w(r)*u1(r) )/r**2
            - 12*( 2*np.sqrt(2)*u(r)*w(r) - w(r)**2 )/r**3
            )
    c_piece = 3*mN**2/k**2 * cN(k) * jn(2,kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            )
    intd = A_piece + c_piece
    return intd

def _cT2_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN):
    kfm = k/hbar
    intd = 3*AN(k)/(mN**2*k**2)*hbar**4 * jn(2,kfm*r/2)*(
            (
                2*np.sqrt(2)*( w(r)*u3(r) - u1(r)*w2(r) )
                - w(r)*w3(r) + w1(r)*w2(r)
                ) / r
            + 2*np.sqrt(2)*( u(r)*w2(r) - w(r)*u2(r) ) / r**2
            + 12*np.sqrt(2)*w(r)*u1(r) / r**3
            - 12*( np.sqrt(2)*u(r)*w(r) + w(r)**2 ) / r**4
            )
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
    integral = quad_vec(_AU_integrand, 0, np.inf,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _AT(k, u, w, AN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, 0, np.inf,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _DU(k, u, w, u1, w1, u2, w2, AN, JN, DN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8)[0]
    return integral * 2

def _DT1(k, u, w, u1, w1, u2, w2, AN, DN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, DN),
                        workers=8)[0]
    return integral * 2

def _DT2(k, u, w, u1, w1, u2, w2, AN, JN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN),
                        workers=8)[0]
    return integral * 2

def _cU(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cU_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8)[0]
    return integral * 2

def _cT1(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cT1_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8)[0]
    return integral * 2

def _cT2(k, u, w, u1, w1, u2, w2, u3, w3, AN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cT2_integrand, 0, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN),
                        workers=8)[0]
    return integral * 2

def _J(k, u, w, AN, JN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_J_integrand, 0, np.inf,
                        args=(k, u, w, AN, JN),
                        workers=8)[0]
    return integral * 2

def _S(k, u, w, SN):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, 0, np.inf,
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

# deuteron.py
# Created 2025.09.30 by Adam Freese
# contributions from both Adam Freese and Alan Sosa
#
# This file contains formulas for form factors as given in the work by
# Wim Cosyn, Adam Freese and Alan Sosa.

import numpy as np
from numpy import sqrt

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec

from ..constants import mN, hbar, mNfm

# Import wave function chooser
from ..wf.chooser import choose_wf

# Import nucleon form factor chooser
from .nucleon.chooser import choose_nff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the form factors
# Add optional wf parameter (last arg) that overrides u/w function arguments if provided.

def AU(k, wf='av18', nff='ba'):
    ''' The EMT form factor AU.
    ----------
    Input:
        - k : float or numpy.array
            float one-dimensional array of k values in GeV
        - wf : string
            Deuteron wave function to use
            Available: av18, paris, cdbonn
            Default: av18
        - nff : string
            Nucleon EMT form factors to use
            Available: ba, mab, hz, point
            Default: ba
    Output:
        numpy.array with form factor values
    Notes:
        Some form factors have multiple options for formulas.
        These are meant for consistency checks.
        The default formula in each case is the fastest to evaluate.
        Form factors with a 'formula' option are:
            - cU
            - cT1
            - cT2
        The options for the formula are 'fast' (default) or 'paper'.
        The latter uses the formula explicitly given in the paper.
        The option is given only to demonstrate that the results are the same,
        but the 'paper' formula is significantly slower.
    '''
    u, w, *_ = choose_wf(wf)
    AN, *_ = choose_nff(nff)
    return _AU(k, u=u, w=w, AN=AN)

def AT(k, wf='av18', nff='ba'):
    ''' The EMT form factor AT.
    See docstring of AU for more info.
    '''
    u, w, *_ = choose_wf(wf)
    AN, *_ = choose_nff(nff)
    return _AT(k, u=u, w=w, AN=AN)

def DU(k, wf='av18', nff='ba'):
    ''' The EMT form factor DU.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, DN, *_ = choose_nff(nff)
    return _DU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT1(k, wf='av18', nff='ba'):
    ''' The EMT form factor DT1.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, DN, *_ = choose_nff(nff)
    return _DT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT2(k, wf='av18', nff='ba'):
    ''' The EMT form factor DT2.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, *_ = choose_nff(nff)
    return _DT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN)

def cU(k, wf='av18', nff='ba', formula='fast'):
    ''' The EMT form factor cU.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    AN, _, _, cN, _ = choose_nff(nff)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin = 1e-2
    return _cU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, cN=cN, rmin=rmin, formula=formula)

def cT1(k, wf='av18', nff='ba', formula='fast'):
    ''' The EMT form factor cT1.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    AN, JN, _, cN, _ = choose_nff(nff)
    return _cT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, JN=JN, cN=cN, formula=formula)

def cT2(k, wf='av18', nff='ba', formula='fast'):
    ''' The EMT form factor cT2.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    AN, JN, *_ = choose_nff(nff)
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin =  1e-2
    return _cT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, JN=JN, rmin=rmin, formula=formula)

def J(k, wf='av18', nff='ba'):
    ''' The EMT form factor J.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, *_ = choose_wf(wf)
    AN, JN, *_ = choose_nff(nff)
    return _J(k, u=u, w=w, u1=u1, w1=w1, AN=AN, JN=JN)

def S(k, wf='av18', nff='ba'):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    u, w, *_ = choose_wf(wf)
    _, _, _, _, SN = choose_nff(nff)
    return _S(k, u=u, w=w, SN=SN)

def sbar(k, wf='av18', nff='ba'):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    _, _, _, _, SN = choose_nff(nff)
    return _sbar(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2,u3=u3,w3=w3, SN=SN)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs:
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
    # TODO: nicer formatting
    kfm = k/hbar
    intd = 24*mNfm**2*(
            kfm**2*r**2*(2*sqrt(2)*u(r) - w(r))*DN(k)*jn(2, kfm*r/2)*w(r)
            +
            4*kfm*r*(sqrt(2)*r*(u(r)*w1(r) - u1(r)*w(r)) + (-2*sqrt(2)*u(r) + w(r))*w(r))*JN(k)*jn(3, kfm*r/2)
            +
            2*(
                r**2*(sqrt(2)*u(r)*w2(r) - 2*sqrt(2)*u1(r)*w1(r) + sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                + r*(-5*sqrt(2)*u(r)*w1(r) + 3*sqrt(2)*u1(r)*w(r) + w(r)*w1(r)) + 6*(2*sqrt(2)*u(r) - w(r))*w(r)
                )*AN(k)*jn(4, kfm*r/2)
            )/(kfm**4*r**2)
    return intd

def _DT2_integrand(r, k, u, w, u1, w1, u2, w2, AN, JN):
    # TODO: nicer formatting
    kfm = k/hbar
    intd = 6*(
            kfm*r*(
                sqrt(2)*r**2*(-u(r)*w2(r) + u2(r)*w(r))
                + 4*r*(sqrt(2)*u1(r) + w1(r))*w(r)
                + 2*(sqrt(2)*u(r) - 2*w(r))*w(r)
                )*JN(k)*jn(2, kfm*r/2)
            -
            2*(
                r**2*(-2*sqrt(2)*u1(r)*w1(r) + 2*sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                + r*(2*sqrt(2)*u(r) + 5*w(r))*w1(r) - 18*w(r)**2
                )*AN(k)*jn(3, kfm*r/2)
            )/(kfm**3*r**3)
    return intd

def _cU_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    # TODO: nicer formatting
    kfm = k/hbar
    intd = (
            6*mNfm**2*r**2*(u(r)**2 + w(r)**2)*cN(k)*jn(0, kfm*r/2)
            +
            (
                -2*(
                    r**2*(u(r)*u2(r) - u1(r)**2 + w(r)*w2(r) - w1(r)**2)
                    - r*(u(r)*u1(r) + w(r)*w1(r))
                    + 2*u(r)**2 + 8*w(r)**2
                    )*jn(2, kfm*r/2)
                +
                (
                    r**2*(u(r)*u2(r) - u1(r)**2 + w(r)*w2(r) - w1(r)**2)
                    + 2*r*(u(r)*u1(r) + w(r)*w1(r))
                    - u(r)**2 - 13*w(r)**2
                    )*jn(0, kfm*r/2)
                )*AN(k)
            )/(12*mNfm**2*r**2)
    return intd

def _cT1_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN):
    # TODO: nicer formatting
    kfm = k/hbar
    intd = (
            42*kfm*r*(
                mNfm**2*r**2*(2*sqrt(2)*u(r) - w(r))*cN(k)*w(r)
                +
                sqrt(2)*(r**2*(-u(r)*w2(r) + u2(r)*w(r)) + 6*u(r)*w(r))*JN(k)
                )*jn(2, kfm*r/2)
            -
            (
                kfm*r*(
                    (
                        11*r**2*(-sqrt(2)*u(r)*w2(r) + 2*sqrt(2)*u1(r)*w1(r) - sqrt(2)*u2(r)*w(r) + w(r)*w2(r) - w1(r)**2)
                        + 2*r*(-11*sqrt(2)*u(r)*w1(r) + sqrt(2)*u1(r)*w(r) + 5*w(r)*w1(r)) + 2*(32*sqrt(2)*u(r) + 5*w(r))*w(r)
                        )*jn(2, kfm*r/2)
                    + 10*(
                        r**2*(sqrt(2)*u(r)*w2(r) - 2*sqrt(2)*u1(r)*w1(r) + sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                        + r*(-5*sqrt(2)*u(r)*w1(r) + 3*sqrt(2)*u1(r)*w(r) + w(r)*w1(r)) + 6*(2*sqrt(2)*u(r) - w(r))*w(r)
                        )*jn(4, kfm*r/2))
                +
                28*(
                    r**2*(-2*sqrt(2)*u1(r)*w1(r) + 2*sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                    + r*(2*sqrt(2)*u(r) + 5*w(r))*w1(r) - 18*w(r)**2
                    )*jn(3, kfm*r/2)
                )*AN(k)
            )/(7*kfm**3*r**3)
    return intd

def _cT2_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    # TODO: nicer formatting
    kfm = k/hbar
    intd = (
            -30*sqrt(2)*kfm*r*(r**2*(-u(r)*w2(r) + u2(r)*w(r)) + 6*u(r)*w(r))*JN(k)*jn(2, kfm*r/2)
            +
            (
                kfm*r*(
                    2*(
                        r**2*(sqrt(2)*u(r)*w2(r) - 2*sqrt(2)*u1(r)*w1(r) + sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                        + r*(-5*sqrt(2)*u(r)*w1(r) + 3*sqrt(2)*u1(r)*w(r) + w(r)*w1(r))
                        + 6*(2*sqrt(2)*u(r) - w(r))*w(r)
                        )*jn(4, kfm*r/2)
                    -
                    (
                        2*r**2*(sqrt(2)*u(r)*w2(r) - 2*sqrt(2)*u1(r)*w1(r) + sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                        + 2*r*(5*sqrt(2)*u(r)*w1(r) - 7*sqrt(2)*u1(r)*w(r) + w(r)*w1(r))
                        + (14*sqrt(2)*u(r) + 23*w(r))*w(r)
                        )*jn(0, kfm*r/2)
                    )
                +
                20*(
                    r**2*(-2*sqrt(2)*u1(r)*w1(r) + 2*sqrt(2)*u2(r)*w(r) - w(r)*w2(r) + w1(r)**2)
                    + r*(2*sqrt(2)*u(r) + 5*w(r))*w1(r)
                    - 18*w(r)**2
                    )*jn(3, kfm*r/2)
             )*AN(k)
            )/(40*kfm*mNfm**2*r**3)
    return intd

def _J_integrand(r, k, u, w, u1, w1, AN, JN):
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

def _sbar_integrand(r, k, u, w, u1, w1, u2, w2, SN):
    kfm = k/hbar
    intd = 6*np.sqrt(2)*SN(k)/kfm**2*jn(2,kfm*r/2)*(
            u(r)*w2(r) - u2(r)*w(r) - 6*u(r)*w(r)/r**2
            )
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The formulas for cbarU, cbarT1 and cbarT2 given in the paper

def _cU_integrand_paper(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, cN):
    kfm = k/hbar
    A_piece = AN(k)*jn(1, kfm*r/2)/(2*mNfm**2*kfm) * (
            u1(r)*u2(r) + w1(r)*w2(r)
            - u(r)*u3(r) - w(r)*w3(r) - 12*w(r)**2/r**3
            )
    c_piece = cN(k)*jn(0, kfm*r/2) * (u(r)**2 + w(r)**2)/2
    return A_piece + c_piece

def _cT1_integrand_paper(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN):
    kfm = k/hbar
    A_piece = 6*AN(k)/kfm**3*jn(3,kfm*r/2)*(
            np.sqrt(2)*(u1(r)*w2(r) + w1(r)*u2(r) - u(r)*w3(r) - w(r)*u3(r))
            + w(r)*w3(r) - w1(r)*w2(r)
            + 2*np.sqrt(2)*(u(r)*w2(r)-w(r)*u2(r))/r
            + 6*np.sqrt(2)*(u(r)*w1(r)-w(r)*u1(r))/r**2
            -12*(2*np.sqrt(2)*u(r)*w(r)-w(r)**2)/r**3
            )
    J_piece = 6*np.sqrt(2)*JN(k)*jn(2,kfm*r/2)/kfm**2*(
            w(r)*u2(r) - u(r)*w2(r) + 6*u(r)*w(r)/r**2
            )
    c_piece = 6*mN**2/k**2 * cN(k) * jn(2,kfm*r/2)*(
            2*np.sqrt(2)*u(r)*w(r) - w(r)**2
            )
    return A_piece + J_piece + c_piece

def _cT2_integrand_paper(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    kfm = k/hbar
    A_piece = 3*AN(k)/(mNfm**2*kfm**2)*jn(2,kfm*r/2)*(
            (2*np.sqrt(2)*(w(r)*u3(r)-u1(r)*w2(r))-w(r)*w3(r)+w1(r)*w2(r))/r
            +(2*np.sqrt(2)*(u(r)*w2(r)-w(r)*u2(r)))/r**2
            +12*np.sqrt(2)*w(r)*u1(r)/r**3
            -12*(np.sqrt(2)*u(r)*w(r)+w(r)**2)/r**4
            )
    J_piece = 3*np.sqrt(2)/(4*mNfm**2)*jn(2,kfm*r/2)*JN(k)*(
            u(r)*w2(r) - w(r)*u2(r) - 6*u(r)*w(r)/r**2
            )
    return A_piece + J_piece

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs:
# 2. Integration
#    quad_vec achieves good speed for parallel calculation of form factors
#    at multiple k values. It's also parallelizable.

def _AU(k, u, w, AN, rmin=0, rmax=np.inf):
    integral = quad_vec(_AU_integrand, rmin, rmax,
                        args=(k,u,w,AN),
                        workers=8
                        )[0]
    return integral * 2

def _AT(k, u, w, AN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, rmin, rmax,
                        args=(k,u,w,AN),
                        workers=8
                        )[0]
    return integral * 2

def _DU(k, u, w, u1, w1, u2, w2, AN, JN, DN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8
                        )[0]
    return integral * 2

def _DT1(k, u, w, u1, w1, u2, w2, AN, JN, DN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8
                        )[0]
    return integral * 2

def _DT2(k, u, w, u1, w1, u2, w2, AN, JN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN),
                        workers=8
                        )[0]
    return integral * 2

def _cU(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cU_integrand
    elif(formula=='paper'):
        integrand = _cU_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8
                        )[0]
    return integral * 2

def _cT1(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cT1_integrand
    elif(formula=='paper'):
        integrand = _cT1_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN),
                        workers=8
                        )[0]
    return integral * 2

def _cT2(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cT2_integrand
    elif(formula=='paper'):
        integrand = _cT2_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN),
                        workers=8
                        )[0]
    return integral * 2

def _J(k, u, w, u1, w1, AN, JN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_J_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, AN, JN),
                        workers=8
                        )[0]
    return integral * 2

def _S(k, u, w, SN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, rmin, rmax,
                        args=(k, u, w, SN),
                        workers=8
                        )[0]
    return integral * 2

def _sbar(k, u, w, u1, w1, u2, w2, u3, w3,SN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_sbar_integrand, rmin, np.inf,
                        args=( k, u, w,u1,w1 ,u2,w2,SN),
                        workers=8
                        )[0]
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

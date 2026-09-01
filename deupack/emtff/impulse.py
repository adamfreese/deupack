# impulse.py
# Created 2025.09.30 by Adam Freese
# contributions from both Adam Freese and Alan Sosa
# migrated from deuteron.py to impulse.py on 2026.05.08
#
# This file contains formulas for form factors as given in the work by
# Wim Cosyn, Adam Freese and Alan Sosa.

import numpy as np
from numpy import sqrt

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec

from ..constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the impulse approximation form factors
# More friendly user interfaces are given in deuteron.py

def AU(k, dwf, nff, rmin=0, rmax=np.inf):
    integral = quad_vec(_AU_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def AT(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def DU(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def DT1(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def DT2(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def cU(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cU_integrand
    elif(formula=='paper'):
        integrand = _cU_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def cT1(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cT1_integrand
    elif(formula=='paper'):
        integrand = _cT1_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def cT2(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='fast'):
        integrand = _cT2_integrand
    elif(formula=='paper'):
        integrand = _cT2_integrand_paper
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def J(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_J_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def S(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def sbar(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_sbar_integrand, rmin, np.inf,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs: Integrands
# Parallelization of the integration requires the integrands to be defined
# as top-level (rather than nested) functions.

def _AU_integrand(r, k, dwf, nff):
    kfm = k/hbar
    intd = nff.AN(k) * (dwf.u(r)**2 + dwf.w(r)**2)/2 * jn(0, kfm*r/2)
    return intd

def _AT_integrand(r, k, dwf, nff):
    kfm = k/hbar
    intd = nff.AN(k) * jn(2, kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            ) * 6*dwf.mN**2/k**2
    return intd

def _DU_integrand(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = 4*nff.AN(k)/kfm**2*jn(2,kfm*r/2)*(
            (2*dwf.u(r)**2+8*dwf.w(r)**2)/r**2
            - (dwf.u(r)*dwf.u1(r)+dwf.w(r)*dwf.w1(r))/r
            + dwf.u(r)*dwf.u2(r) + dwf.w(r)*dwf.w2(r)
            - dwf.u1(r)**2 - dwf.w1(r)**2
            )
    J_piece = -12*nff.JN(k)/kfm*jn(1,kfm*r/2)*dwf.w(r)**2/r
    D_piece = 2*nff.DN(k)*jn(0,kfm*r/2)*(dwf.u(r)**2 + dwf.w(r)**2)
    intd = A_piece + J_piece + D_piece
    return intd

def _DT1_integrand(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = 48*dwf.mNfm**2*nff.AN(k)/kfm**4*jn(4,kfm*r/2)*(
            sqrt(2)*(dwf.u(r)*dwf.w2(r) + dwf.w(r)*dwf.u2(r)
                     - 2*dwf.u1(r)*dwf.w1(r))
            - dwf.w(r)*dwf.w2(r) + dwf.w1(r)**2
            + (sqrt(2)*(3*dwf.w(r)*dwf.u1(r) - 5*dwf.u(r)*dwf.w1(r))
               + dwf.w(r)*dwf.w1(r))/r
            + 6*(2*sqrt(2)*dwf.u(r) - dwf.w(r))*dwf.w(r)/r**2
            )
    J_piece = 96*dwf.mNfm**2*nff.JN(k)/kfm**3*jn(3,kfm*r/2)*(
            sqrt(2)*(dwf.u(r)*dwf.w1(r) - dwf.w(r)*dwf.u1(r))
            - (2*sqrt(2)*dwf.u(r) - dwf.w(r))*dwf.w(r)/r
            )
    D_piece = 24*dwf.mNfm**2*nff.DN(k)/kfm**2*jn(2,kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            )
    intd = A_piece + J_piece + D_piece
    return intd

def _DT2_integrand(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = 12*nff.AN(k)/kfm**3*jn(3,kfm*r/2)*(
            ( (2*sqrt(2)*dwf.u2(r) - dwf.w2(r))*dwf.w(r)
                - (2*sqrt(2)*dwf.u1(r) - dwf.w1(r))*dwf.w1(r) )/r
            + (2*sqrt(2)*dwf.u(r) + 5*dwf.w(r))*dwf.w1(r)/r**2
            - 18*dwf.w(r)**2/r**3
            )
    J_piece = 6*nff.JN(k)/kfm**2*jn(2,kfm*r/2)*(
            sqrt(2)*(dwf.u(r)*dwf.w2(r) - dwf.w(r)*dwf.u2(r))
            - 4*(sqrt(2)*dwf.u1(r) + dwf.w1(r))*dwf.w(r)/r
            - (2*sqrt(2)*dwf.u(r) - dwf.w(r))*dwf.w(r)/r**2
            + 3*dwf.w(r)**2/r**3
            )
    intd = A_piece + J_piece
    return intd

def _cU_integrand(r, k, dwf, nff):
    # Uses IBP to get rid of third derivatives of u(r) and w(r)
    kfm = k/hbar
    A_piece_1 = nff.AN(k)/(dwf.mNfm**2*kfm)*jn(1,kfm*r/2)*(
            dwf.u1(r)*dwf.u2(r) - dwf.u(r)*dwf.u2(r)/r
            + dwf.w1(r)*dwf.w2(r) - dwf.w(r)*dwf.w2(r)/r
            - 6*dwf.w(r)**2/r**3
            )
    A_piece_0 = nff.AN(k)/(4*dwf.mNfm**2)*jn(0,kfm*r/2)*(
            dwf.u(r)*dwf.u2(r) + dwf.w(r)*dwf.w2(r)
            )
    c_piece = nff.cN(k)/2*jn(0,kfm*r/2)*(
            dwf.u(r)**2 + dwf.w(r)**2
            )
    intd = A_piece_1 + A_piece_0 + c_piece
    return intd

def _cT1_integrand(r, k, dwf, nff):
    # Uses IBP to get rid of third derivatives of u(r) and w(r)
    kfm = k/hbar
    A_piece_3 = 12*nff.AN(k)/kfm**3*jn(3,kfm*r/2)*(
            sqrt(2)*(dwf.u1(r)*dwf.w2(r) + dwf.w1(r)*dwf.u2(r))
            - dwf.w1(r)*dwf.w2(r)
            - sqrt(2)*(dwf.u(r)*dwf.w2(r) + 3*dwf.w(r)*dwf.u2(r))/r
            + 2*dwf.w(r)*dwf.w2(r)/r
            + 3*sqrt(2)*(dwf.u(r)*dwf.w1(r)-dwf.w(r)*dwf.u1(r))/r**2
            - 6*(2*sqrt(2)*dwf.u(r)*dwf.w(r)-dwf.w(r)**2)/r**3
            )
    A_piece_2 = 3*nff.AN(k)/kfm**2*jn(2,kfm*r/2)*(
            + sqrt(2)*(dwf.u(r)*dwf.w2(r) + dwf.w(r)*dwf.u2(r))
            - dwf.w(r)*dwf.w2(r)
            )
    J_piece = 6*sqrt(2)*nff.JN(k)*jn(2,kfm*r/2)/kfm**2*(
            dwf.w(r)*dwf.u2(r) - dwf.u(r)*dwf.w2(r) + 6*dwf.u(r)*dwf.w(r)/r**2
            )
    c_piece = 6*dwf.mN**2/k**2 * nff.cN(k) * jn(2,kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            )
    return A_piece_3 + A_piece_2 + J_piece + c_piece

def _cT2_integrand(r, k, dwf, nff):
    # Uses IBP to get rid of third derivatives of u(r) and w(r)
    kfm = k/hbar
    A_piece_2 = 6*nff.AN(k)/(dwf.mNfm**2*kfm**2)*jn(2,kfm*r/2)*(
            (sqrt(2)*(dwf.u1(r)*dwf.w2(r) + dwf.w1(r)*dwf.u2(r))
             - dwf.w1(r)*dwf.w2(r))/r
            - sqrt(2)*(dwf.u(r)*dwf.w2(r) + 3*dwf.w(r)*dwf.u2(r))/r**2
            + 2*dwf.w(r)*dwf.w2(r)/r**2
            - 6*sqrt(2)*dwf.w(r)*dwf.u1(r)/r**3
            + 6*(sqrt(2)*dwf.u(r)*dwf.w(r) + dwf.w(r)**2)/r**4
            )
    A_piece_1 = 3*nff.AN(k)/(2*dwf.mNfm**2*kfm)*jn(1,kfm*r/2)*(
            2*sqrt(2)*dwf.w(r)*dwf.u2(r)
            - dwf.w(r)*dwf.w2(r)
            )/r
    J_piece = -3*sqrt(2)/(4*dwf.mNfm**2)*jn(2,kfm*r/2)*nff.JN(k)*(
            dwf.u(r)*dwf.w2(r) - dwf.w(r)*dwf.u2(r) - 6*dwf.u(r)*dwf.w(r)/r**2
            )
    return A_piece_2 + A_piece_1 + J_piece

def _J_integrand(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = 9/2*nff.AN(k)/kfm * jn(1,kfm*r/2) * dwf.w(r)**2/r
    J0_piece = nff.JN(k)*jn(0,kfm*r/2)*(dwf.u(r)**2 - dwf.w(r)**2/2)
    J2_piece = nff.JN(k)*jn(2,kfm*r/2)*(dwf.w(r)**2 + sqrt(2)*dwf.u(r)*dwf.w(r))/2
    intd = A_piece + J0_piece + J2_piece
    return intd

def _S_integrand(r, k, dwf, nff):
    kfm = k/hbar
    S0_piece = nff.SN(k)*jn(0,kfm*r/2)*(dwf.u(r)**2 - dwf.w(r)**2/2)
    S2_piece = nff.SN(k)*jn(2,kfm*r/2)*(dwf.w(r)**2 + sqrt(2)*dwf.u(r)*dwf.w(r))/2
    intd = S0_piece + S2_piece
    return intd

def _sbar_integrand(r, k, dwf, nff):
    kfm = k/hbar
    intd = -6*sqrt(2)*nff.SN(k)/kfm**2*jn(2,kfm*r/2)*(
            dwf.u(r)*dwf.w2(r) - dwf.u2(r)*dwf.w(r) - 6*dwf.u(r)*dwf.w(r)/r**2
            )
    return intd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The formulas for cbarU, cbarT1 and cbarT2 given in the paper

def _cU_integrand_paper(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = nff.AN(k)*jn(1, kfm*r/2)/(2*dwf.mNfm**2*kfm) * (
            dwf.u1(r)*dwf.u2(r) + dwf.w1(r)*dwf.w2(r)
            - dwf.u(r)*dwf.u3(r) - dwf.w(r)*dwf.w3(r) - 12*dwf.w(r)**2/r**3
            )
    c_piece = nff.cN(k)*jn(0, kfm*r/2) * (dwf.u(r)**2 + dwf.w(r)**2)/2
    return A_piece + c_piece

def _cT1_integrand_paper(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = 6*nff.AN(k)/kfm**3*jn(3,kfm*r/2)*(
            sqrt(2)*(dwf.u1(r)*dwf.w2(r) + dwf.w1(r)*dwf.u2(r)
                        - dwf.u(r)*dwf.w3(r) - dwf.w(r)*dwf.u3(r))
            + dwf.w(r)*dwf.w3(r) - dwf.w1(r)*dwf.w2(r)
            + 2*sqrt(2)*(dwf.u(r)*dwf.w2(r)-dwf.w(r)*dwf.u2(r))/r
            + 6*sqrt(2)*(dwf.u(r)*dwf.w1(r)-dwf.w(r)*dwf.u1(r))/r**2
            -12*(2*sqrt(2)*dwf.u(r)*dwf.w(r)-dwf.w(r)**2)/r**3
            )
    J_piece = 6*sqrt(2)*nff.JN(k)*jn(2,kfm*r/2)/kfm**2*(
            dwf.w(r)*dwf.u2(r) - dwf.u(r)*dwf.w2(r) + 6*dwf.u(r)*dwf.w(r)/r**2
            )
    c_piece = 6*dwf.mN**2/k**2 * nff.cN(k) * jn(2,kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            )
    return A_piece + J_piece + c_piece

def _cT2_integrand_paper(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = -3*nff.AN(k)/(dwf.mNfm**2*kfm**2)*jn(2,kfm*r/2)*(
            (2*sqrt(2)*(dwf.w(r)*dwf.u3(r)-dwf.u1(r)*dwf.w2(r))
             -dwf.w(r)*dwf.w3(r)+dwf.w1(r)*dwf.w2(r))/r
            +(2*sqrt(2)*(dwf.u(r)*dwf.w2(r)-dwf.w(r)*dwf.u2(r)))/r**2
            +12*sqrt(2)*dwf.w(r)*dwf.u1(r)/r**3
            -12*(sqrt(2)*dwf.u(r)*dwf.w(r)+dwf.w(r)**2)/r**4
            )
    J_piece = -3*sqrt(2)/(4*dwf.mNfm**2)*jn(2,kfm*r/2)*nff.JN(k)*(
            dwf.u(r)*dwf.w2(r) - dwf.w(r)*dwf.u2(r) - 6*dwf.u(r)*dwf.w(r)/r**2
            )
    return A_piece + J_piece

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

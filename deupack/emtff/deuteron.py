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

# Default wave function is an AV18 instance
from ..wf.av18 import dwf_av18
wf_default = dwf_av18()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the form factors

def AU(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor AU.
    ----------
    Input:
        - k : float or numpy.array
            float one-dimensional array of k values in GeV
        - wf : DWF or string
            Deuteron wave function to use
            See wf.chooser.choose_wf for available options
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
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _AU(k, dwf=dwf, nff=_nff)

def AT(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor AT.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _AT(k, dwf=dwf, nff=_nff)

def DU(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor DU.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _DU(k, dwf=dwf, nff=_nff)

def DT1(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor DT1.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _DT1(k, dwf=dwf, nff=_nff)

def DT2(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor DT2.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _DT2(k, dwf=dwf, nff=_nff)

def cU(k, wf=wf_default, nff='ba', formula='fast'):
    ''' The EMT form factor cU.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin = 1e-2
    return _cU(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)

def cT1(k, wf=wf_default, nff='ba', formula='fast'):
    ''' The EMT form factor cT1.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _cT1(k, dwf=dwf, nff=_nff, formula=formula)

def cT2(k, wf=wf_default, nff='ba', formula='fast'):
    ''' The EMT form factor cT2.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    _nff = choose_nff(nff)
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin =  1e-2
    return _cT2(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)

def J(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor J.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _J(k, dwf=dwf, nff=_nff)

def S(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _S(k, dwf=dwf, nff=_nff)

def sbar(k, wf=wf_default, nff='ba'):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return _sbar(k, dwf=dwf, nff=_nff)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs:
# 1. Integrands
#    Parallelization of the integration requires the integrands to be defined
#    as top-level (rather than nested) functions.

def _AU_integrand(r, k, dwf, nff):
    kfm = k/hbar
    intd = nff.AN(k) * (dwf.u(r)**2 + dwf.w(r)**2)/2 * jn(0, kfm*r/2)
    return intd

def _AT_integrand(r, k, dwf, nff):
    kfm = k/hbar
    intd = nff.AN(k) * jn(2, kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            ) * 6*mN**2/k**2
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
    A_piece = 48*mNfm**2*nff.AN(k)/kfm**4*jn(4,kfm*r/2)*(
            sqrt(2)*(dwf.u(r)*dwf.w2(r) + dwf.w(r)*dwf.u2(r)
                     - 2*dwf.u1(r)*dwf.w1(r))
            - dwf.w(r)*dwf.w2(r) + dwf.w1(r)**2
            + (sqrt(2)*(3*dwf.w(r)*dwf.u1(r) - 5*dwf.u(r)*dwf.w1(r))
               + dwf.w(r)*dwf.w1(r))/r
            + 6*(2*sqrt(2)*dwf.u(r) - dwf.w(r))*dwf.w(r)/r**2
            )
    J_piece = 96*mNfm**2*nff.JN(k)/kfm**3*jn(3,kfm*r/2)*(
            sqrt(2)*(dwf.u(r)*dwf.w1(r) - dwf.w(r)*dwf.u1(r))
            - (2*sqrt(2)*dwf.u(r) - dwf.w(r))*dwf.w(r)/r
            )
    D_piece = 24*mNfm**2*nff.DN(k)/kfm**2*jn(2,kfm*r/2)*(
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
    A_piece_1 = nff.AN(k)/(mNfm**2*kfm)*jn(1,kfm*r/2)*(
            dwf.u1(r)*dwf.u2(r) - dwf.u(r)*dwf.u2(r)/r
            + dwf.w1(r)*dwf.w2(r) - dwf.w(r)*dwf.w2(r)/r
            - 6*dwf.w(r)**2/r**3
            )
    A_piece_0 = nff.AN(k)/(4*mNfm**2)*jn(0,kfm*r/2)*(
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
    c_piece = 6*mN**2/k**2 * nff.cN(k) * jn(2,kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            )
    return A_piece_3 + A_piece_2 + J_piece + c_piece

def _cT2_integrand(r, k, dwf, nff):
    # Uses IBP to get rid of third derivatives of u(r) and w(r)
    kfm = k/hbar
    A_piece_2 = 6*nff.AN(k)/(mNfm**2*kfm**2)*jn(2,kfm*r/2)*(
            (sqrt(2)*(dwf.u1(r)*dwf.w2(r) + dwf.w1(r)*dwf.u2(r))
             - dwf.w1(r)*dwf.w2(r))/r
            - sqrt(2)*(dwf.u(r)*dwf.w2(r) + 3*dwf.w(r)*dwf.u2(r))/r**2
            + 2*dwf.w(r)*dwf.w2(r)/r**2
            - 6*sqrt(2)*dwf.w(r)*dwf.u1(r)/r**3
            + 6*(sqrt(2)*dwf.u(r)*dwf.w(r) + dwf.w(r)**2)/r**4
            )
    A_piece_1 = 3*nff.AN(k)/(2*mNfm**2*kfm)*jn(1,kfm*r/2)*(
            2*sqrt(2)*dwf.w(r)*dwf.u2(r)
            - dwf.w(r)*dwf.w2(r)
            )/r
    J_piece = -3*sqrt(2)/(4*mNfm**2)*jn(2,kfm*r/2)*nff.JN(k)*(
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
    A_piece = nff.AN(k)*jn(1, kfm*r/2)/(2*mNfm**2*kfm) * (
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
    c_piece = 6*mN**2/k**2 * nff.cN(k) * jn(2,kfm*r/2)*(
            2*sqrt(2)*dwf.u(r)*dwf.w(r) - dwf.w(r)**2
            )
    return A_piece + J_piece + c_piece

def _cT2_integrand_paper(r, k, dwf, nff):
    kfm = k/hbar
    A_piece = -3*nff.AN(k)/(mNfm**2*kfm**2)*jn(2,kfm*r/2)*(
            (2*sqrt(2)*(dwf.w(r)*dwf.u3(r)-dwf.u1(r)*dwf.w2(r))
             -dwf.w(r)*dwf.w3(r)+dwf.w1(r)*dwf.w2(r))/r
            +(2*sqrt(2)*(dwf.u(r)*dwf.w2(r)-dwf.w(r)*dwf.u2(r)))/r**2
            +12*sqrt(2)*dwf.w(r)*dwf.u1(r)/r**3
            -12*(sqrt(2)*dwf.u(r)*dwf.w(r)+dwf.w(r)**2)/r**4
            )
    J_piece = -3*sqrt(2)/(4*mNfm**2)*jn(2,kfm*r/2)*nff.JN(k)*(
            dwf.u(r)*dwf.w2(r) - dwf.w(r)*dwf.u2(r) - 6*dwf.u(r)*dwf.w(r)/r**2
            )
    return A_piece + J_piece

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs:
# 2. Integration
#    quad_vec achieves good speed for parallel calculation of form factors
#    at multiple k values. It's also parallelizable.

def _AU(k, dwf, nff, rmin=0, rmax=np.inf):
    integral = quad_vec(_AU_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _AT(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _DU(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _DT1(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _DT2(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _cU(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
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

def _cT1(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
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

def _cT2(k, dwf, nff, rmin=0, rmax=np.inf, formula='fast'):
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

def _J(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_J_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _S(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, rmin, rmax,
                        args=(k, dwf, nff),
                        workers=8
                        )[0]
    return integral * 2

def _sbar(k, dwf, nff, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_sbar_integrand, rmin, np.inf,
                        args=(k, dwf, nff),
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

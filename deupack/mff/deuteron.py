# deuteron.py
# Created 2025.09.30 by Adam Freese
# contributions from both Adam Freese and Alan Sosa
#
# This file contains formulas for form factors as given in the work by
# Wim Cosyn, Adam Freese and Alan Sosa.

import numpy as np

from scipy.special import spherical_jn as jn
from scipy.integrate import quad_vec

from ..constants import mN, hbar

# Import wave function chooser
from ..wf.chooser import choose_wf

# Import nucleon form factor chooser
from .nucleon.chooser import choose_nff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the form factors
# Add optional wf parameter (last arg) that overrides u/w function arguments if provided.

def AU(k, wf='av18', nff='mit'):
    ''' The mechanical form factor AU.

    k should be a float or numpy array of momentum transfer values in GeV.
    k=0 will automatically be pushed to 1e-6 to avoid division by zero.

    Pass wf='av18' or wf='paris' to select wavefunction.

    Pass nff='ba', 'mab', 'hz' or 'mit' to select nucleon form factors.
    '''
    u, w, *_ = choose_wf(wf)
    AN, *_ = choose_nff(nff)
    return _AU(k, u=u, w=w, AN=AN)

def AT(k, wf='av18', nff='mit'):
    ''' The mechanical form factor AT.
    See docstring of AU for more info.
    '''
    u, w, *_ = choose_wf(wf)
    AN, *_ = choose_nff(nff)
    return _AT(k, u=u, w=w, AN=AN)

def DU(k, wf='av18', nff='mit'):
    ''' The mechanical form factor DU.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, DN, *_ = choose_nff(nff)
    return _DU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT1(k, wf='av18', nff='mit'):
    ''' The mechanical form factor DT1.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, DN, *_ = choose_nff(nff)
    return _DT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN, DN=DN)

def DT2(k, wf='av18', nff='mit'):
    ''' The mechanical form factor DT2.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, _, _ = choose_wf(wf)
    AN, JN, *_ = choose_nff(nff)
    return _DT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, AN=AN, JN=JN)

def cU(k, wf='av18', nff='mit'):
    ''' The mechanical form factor cU.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    AN, _, _, cN, _ = choose_nff(nff)
    # Need to change rmin from 0 to 1e-2 for the Paris wf,
    # because of an instability at small r
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin = 1e-2
    return _cU(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, cN=cN, rmin=rmin)

def cT1(k, wf='av18', nff='mit', formula='cT1'):
    ''' The mechanical form factor cT1.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    AN, JN, _, cN, _ = choose_nff(nff)
    return _cT1(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, JN=JN, cN=cN, formula=formula)

def cT2(k, wf='av18', nff='mit', formula='cT2'):
    ''' The mechanical form factor cT2.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    # Need to change rmin from 0 to 1e-2 for the Paris wf,
    # because of an instability at small r
    AN, JN, *_ = choose_nff(nff)
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin =  1e-2
    return _cT2(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2, u3=u3, w3=w3, AN=AN, JN=JN, rmin=rmin, formula=formula)

def J(k, wf='av18', nff='mit', formula='form1'):
    ''' The mechanical form factor J.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, *_ = choose_wf(wf)
    AN, JN, *_ = choose_nff(nff)
    return _J(k, u=u, w=w, u1=u1, w1=w1, AN=AN, JN=JN, formula=formula)

def S(k, wf='av18', nff='mit'):
    ''' The mechanical form factor S.
    See docstring of AU for more info.
    '''
    u, w, *_ = choose_wf(wf)
    _, _, _, _, SN = choose_nff(nff)
    return _S(k, u=u, w=w, SN=SN)

def sbar(k, wf='av18', nff='mit', formula='sbar'):
    ''' The mechanical form factor S.
    See docstring of AU for more info.
    '''
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    _, _, _, _, SN = choose_nff(nff)
    return _sbar(k, u=u, w=w, u1=u1, w1=w1, u2=u2, w2=w2,u3=u3,w3=w3, SN=SN, formula=formula)


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
    A_piece = -12/kfm**3 * AN(k) * jn(3,kfm*r/2)*(
            ( (2*np.sqrt(2)*u2(r)-w2(r))*w(r) - (2*np.sqrt(2)*u1(r)-w1(r))*w1(r)) / r
            + ( 2*np.sqrt(2)*u(r)+5*w(r) )*w1(r)/r**2
            - 18*w(r)**2/r**3
            )
    J_piece = -6/kfm**2 * JN(k) * jn(2,kfm*r/2)*(
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

def _cT1_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN):
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
    # TODO: determine whether this J piece is right or too big by a factor 2
    J_piece = 6*np.sqrt(2)*JN(k)*jn(2,kfm*r/2)/kfm**2*(
            w(r)*u2(r) - u(r)*w2(r) + 6*u(r)*w(r)/r**2
            )
    intd = A3_piece + A24_piece + c_piece + J_piece
    return intd

def _cT1_integrandAlan(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN):
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
    # TODO: new J pieces need cross-check
    J3_piece = 3*JN(k)/(5*np.sqrt(2)*r*kfm)*jn(3,kfm*r/2)*(
           3*(r*u1(r)*w(r)-r*w1(r)*u(r)+2*w(r)*u(r))
           )
    J1_piece = 3*JN(k)/(5*np.sqrt(2)*r*kfm)*jn(1,kfm*r/2)*(
           2*(-r*u1(r)*w(r)+r*w1(r)*u(r)+3*w(r)*u(r))
           )
    intd = A3_piece + A24_piece + c_piece + J1_piece + J3_piece
    return intd

def _cT2_integrand(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    kfm = k/hbar
    A2_piece = jn(2,kfm*r/2)*(
            2*(w1(r)*w2(r) - np.sqrt(2)*(w1(r)*u2(r) + u1(r)*w2(r))) / r
            + (2*np.sqrt(2)*u(r) - w(r))*w2(r) / r**2
            + 12*np.sqrt(2)*w(r)*u1(r) / r**3
            - 12*( np.sqrt(2)*u(r)*w(r) + w(r)**2 ) / r**4
            )
    A13_piece = kfm*(2*jn(1,kfm*r/2) - 3*jn(3,kfm*r/2))/10*(
            w2(r) - 2*np.sqrt(2)*u2(r)
            ) * w(r) / r
    intd = 3*AN(k)/(mN**2*k**2)*hbar**4 * (A2_piece + A13_piece)
    J_piece = 3*np.sqrt(2)*JN(k)*jn(2,kfm*r/2)/(2*mN/hbar)**2*(
            u(r)*w2(r) - w(r)*u2(r) - 6*u(r)*w(r)/r**2
            )
    intd += J_piece
    return intd

def _cT2_integrandAlan(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    kfm = k/hbar
    A0_term = jn(0,kfm*r/2)*(
            -7*r**2*w1(r)*(2*np.sqrt(2)*u1(r)-w1(r))
            -7*r*w(r)*(-np.sqrt(2)*r*u2(r)+7*np.sqrt(2)*u1(r)+r*w2(r)-w1(r))
            +7*np.sqrt(2)*r*u(r)*(r*w2(r)+5*w1(r))+49*np.sqrt(2)*u(r)*w(r)+161./2.*w(r)**2
            )
    A2_term = jn(2,kfm*r/2)*(
            5*r**2*w1(r)*(2*np.sqrt(2)*u1(r)-w1(r))
            +5*r*w(r)*(-np.sqrt(2)*r*u2(r)+4*np.sqrt(2)*u1(r)+r*w2(r)-w1(r))
            -5*np.sqrt(2)*r*u(r)*(r*w2(r)+2*w1(r))+10*np.sqrt(2)*u(r)*w(r)+100*w(r)**2
            )
    A4_term = jn(4,kfm*r/2)*(
            12*r**2*w1(r)*(2*np.sqrt(2)*u1(r)-w1(r))
            -144*np.sqrt(2)*u(r)*w(r)+72*w(r)**2
            -12*r*w(r)*(np.sqrt(2)*r*u2(r)+3*np.sqrt(2)*u1(r)-r*w2(r)+w1(r))
            +12*np.sqrt(2)*r*u(r)*(5*w1(r)-r*w2(r))
            )
    intd = -AN(k)/(140*mN**2*r**2)*hbar**2 * (A0_term + A2_term+A4_term)
    # J terms (confirmed)
    J3_piece = -3*JN(k)*k*hbar/(20*np.sqrt(2)*mN**2*r)*jn(3,kfm*r/2)*(
           3*(r*u1(r)*w(r) - r*u(r)*w1(r)+2*u(r)*w(r))
           )
    J1_piece = -3*JN(k)*k*hbar/(20*np.sqrt(2)*mN**2*r)*jn(1,kfm*r/2)*(
           2*(-r*u1(r)*w(r) + r*u(r)*w1(r)+3*u(r)*w(r))
           )
    
    intd += J1_piece + J3_piece
    return intd

def _cT2_integrandAlan3(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    # missing J terms; will deprecate since cross-checks were passed
    kfm = k/hbar
    A2_term = 10.*jn(2,kfm*r/2)/(kfm*r)*(
            r**3*w1(r)*(np.sqrt(2.)*u2(r)-w2(r))+r**3*np.sqrt(2.)*u1(r)*w2(r)
            +12*w(r)**2-r**3*np.sqrt(2)*u(r)*w3(r)+6*r*np.sqrt(2)*u(r)*w1(r)
            -r*w(r)*(6*np.sqrt(2)*u1(r)+np.sqrt(2)*r**2*u3(r)-r**2*w3(r))
            )
    A13_term = (
                ( 2*jn(3,kfm*r/2) - 3*jn(1,kfm*r/2))*np.sqrt(2)*r**2*(
                    u(r)*w2(r) - w(r)*u2(r)
                    )
                + (jn(1,kfm*r/2) - 4*jn(3,kfm*r/2))*(
                    6*np.sqrt(2)*u(r)*w(r)
                    )
                )
    intd = -3*AN(k)/(10.0*mN**2*k*r**3)*hbar**3 * (A2_term + A13_term)
    return intd

def _cT2_integrandAdam3(r, k, u, w, u1, w1, u2, w2, u3, w3, AN, JN):
    # missing J terms; will deprecate since cross-checks were passed
    kfm = k/hbar
    A2_term = jn(2,kfm*r/2)*(
            (2*np.sqrt(2)*(w(r)*u3(r)-u1(r)*w2(r))-w(r)*w3(r)+w1(r)*w2(r))/r
            +(2*np.sqrt(2)*(u(r)*w2(r)-w(r)*u2(r)))/r**2
            +12*np.sqrt(2)*w(r)*u1(r)/r**3
            -12*(np.sqrt(2)*u(r)*w(r)+w(r)**2)/r**4
            )
    intd =  3*AN(k)/(mN**2*k**2)*hbar**4 * (A2_term)
    return intd

def _J_integrand(r, k, u, w, u1, w1, AN, JN):
    kfm = k/hbar
    A_piece = 9/2*AN(k)/kfm * jn(1,kfm*r/2) * w(r)**2/r
    J0_piece = JN(k)*jn(0,kfm*r/2)*(u(r)**2 - w(r)**2/2)
    J2_piece = JN(k)*jn(2,kfm*r/2)*(w(r)**2 + np.sqrt(2)*u(r)*w(r))/2
    intd = A_piece + J0_piece + J2_piece
    return intd

def _J_integrand_alt(r, k, u, w, u1, w1, AN, JN):
    kfm = k/hbar
    A_piece = 9/2*AN(k)/kfm * jn(1,kfm*r/2) * w(r)**2/r
    J_piece = JN(k)*jn(1,kfm*r/2)/kfm*(
            4*w(r)*w1(r) - 4*u(r)*u1(r) + np.sqrt(2)*(u(r)*w1(r)+u1(r)*w(r))
            + (4*u(r)**2 - w(r)**2 + np.sqrt(2)*u(r)*w(r))/r
            )
    intd = A_piece + J_piece
    return intd

def _S_integrand(r, k, u, w, SN):
    kfm = k/hbar
    S0_piece = SN(k)*jn(0,kfm*r/2)*(u(r)**2 - w(r)**2/2)
    S2_piece = SN(k)*jn(2,kfm*r/2)*(w(r)**2 + np.sqrt(2)*u(r)*w(r))/2
    intd = S0_piece + S2_piece
    return intd

def _sbar_integrandAdam(r, k, u, w,u1,w1,u2,w2, SN):
    kfm = k/hbar
    intd = 6*np.sqrt(2)*SN(k)/kfm**2*jn(2,kfm*r/2)*(
            u(r)*w2(r) - u2(r)*w(r) - 6*u(r)*w(r)/r**2
            )
    return intd

def _sbar_integrandAlan(r, k, u, w,u1,w1 ,u2,w2,SN):
    kfm = k/hbar
    J3_piece = 3*SN(k)/(5*np.sqrt(2)*kfm*r)*jn(3,kfm*r/2)*(
           3*(r*u1(r)*w(r) - r*u(r)*w1(r)+2*u(r)*w(r))
           )
    J1_piece = 3*SN(k)/(5*np.sqrt(2)*kfm*r)*jn(1,kfm*r/2)*(
           2*(-r*u1(r)*w(r) + r*u(r)*w1(r)+3*u(r)*w(r))
           )
    intd = J1_piece + J3_piece
    return intd


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the MFFs:
# 2. Integration
#    quad_vec achieves good speed for parallel calculation of form factors
#    at multiple k values. It's also parallelizable.

def _AU(k, u, w, AN, rmin=0, rmax=np.inf):
    integral = quad_vec(_AU_integrand, rmin, rmax,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _AT(k, u, w, AN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_AT_integrand, rmin, rmax,
                        args=(k,u,w,AN),
                        workers=8)[0]
    return integral * 2

def _DU(k, u, w, u1, w1, u2, w2, AN, JN, DN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8)[0]
    return integral * 2

def _DT1(k, u, w, u1, w1, u2, w2, AN, JN, DN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT1_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN, DN),
                        workers=8)[0]
    return integral * 2

def _DT2(k, u, w, u1, w1, u2, w2, AN, JN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DT2_integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, u2, w2, AN, JN),
                        workers=8)[0]
    return integral * 2

def _cU(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cU_integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, cN),
                        workers=8)[0]
    return integral * 2

def _cT1(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN, rmin=0, rmax=np.inf, formula='cT1'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='cT1'):
        integrand = _cT1_integrand
    elif(formula=='cT1Alan'):
        integrand = _cT1_integrandAlan
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, cN),
                        workers=8)[0]
    return integral * 2

def _cT2(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN, rmin=0, rmax=np.inf, formula='cT2'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='cT2'):
        integrand = _cT2_integrand
    elif(formula=='cT2Adam3'):
        integrand = _cT2_integrandAdam3
    elif(formula=='cT2Alan3'):
        integrand = _cT2_integrandAlan3
    elif(formula=='cT2Alan'):
        integrand = _cT2_integrandAlan
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=(k, u, w, u1, w1, u2, w2, u3, w3, AN, JN),
                        workers=8)[0]
    return integral * 2

def _J(k, u, w, u1, w1, AN, JN, rmin=0, rmax=np.inf, formula='form1'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='form1'):
        integrand = _J_integrand
    elif(formula=='form2'):
        integrand = _J_integrand_alt
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, rmax,
                        args=(k, u, w, u1, w1, AN, JN),
                        workers=8)[0]
    return integral * 2

def _S(k, u, w, SN, rmin=0, rmax=np.inf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_S_integrand, rmin, rmax,
                        args=(k, u, w, SN),
                        workers=8)[0]
    return integral * 2


def _sbar(k, u, w, u1, w1, u2, w2, u3, w3,SN, rmin=0, rmax=np.inf, formula='sbar'):
    k = regulate_zero(k) # avoid division by zero
    if(formula=='sbar'):
        integrand = _sbar_integrandAdam
    elif(formula=='sbarAlan'):
        integrand = _sbar_integrandAlan
    else:
        raise ValueError("{} is not a valid formula key.".format(formula))
    integral = quad_vec(integrand, rmin, np.inf,
                        args=( k, u, w,u1,w1 ,u2,w2,SN),
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Temporary stuff for internal cross-checks

def _DT1_zero_integrand(r, u, w, u1, w1, u2, w2):
    term1 = np.sqrt(2)*r**3*( 24*u(r)*w1(r) - 24*w(r)*u1(r) )
    term2 = r**4*(4*np.sqrt(2)*u(r)*w2(r) + 4*np.sqrt(2)*w(r)*u2(r) - 4*w(r)*w2(r))
    term3 = 12*np.sqrt(2)*r**2*u(r)*w(r)
    return (mN/hbar)**2*(term1+term2+term3) / 315

def DT1_zero(wf='av18', nff='mit'):
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    _, _, DN, *_ = choose_nff(nff)
    DN0 = 2*DN(0)
    Qd = AT(0, wf=wf, nff=nff)
    integral = quad_vec(_DT1_zero_integrand, 0, np.inf,
                        args=(u, w, u1, w1, u2, w2),
                        workers=8)[0]
    result = (2*DN0 - 5/7)*Qd + integral
    return result

def _DT2_zero_integrand(r, u, w, u1, w1, u2, w2):
    return -(
            3*w(r)**2
            + 4*r**2*(
                np.sqrt(2)*(u(r)*w2(r) + w(r)*u2(r)) - w(r)*w2(r)
                )
            + 12*np.sqrt(2)*r*u(r)*w1(r)
            ) / 70

def DT2_zero(wf='av18'):
    u, w, u1, w1, u2, w2, u3, w3 = choose_wf(wf)
    integral = quad_vec(_DT2_zero_integrand, 0, np.inf,
                        args=(u, w, u1, w1, u2, w2),
                        workers=8)[0]
    return integral

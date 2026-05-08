# dwf.py
# Created 2026.01.06 by Adam Freese
#
# A base class specifying the interface for deuteron wave functions

import numpy as np
from scipy.integrate import quad

from ..constants import kappa, hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A base class defining functions expected to exist in a deuteron wave function

class DWF:
    ''' An empty class. This just defines the expected interface of any deuteron
    wave funtion. This ensures that an object in any derived class will have
    the expected functions, even if the derived class doesn't define them
    (in which case they will just return 0).
    '''

    def __init__(self):
        self.name = "" # every dwf class should have a name
        self.mN = 0.93891875569 # GeV ... allows per-target override
        self.mNfm = self.mN / hbar
        return

    # The following functions are expected to exist ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        return 0

    def u1(self, r):
        return 0

    def u2(self, r):
        return 0

    def u3(self, r):
        return 0

    def w(self, r):
        return 0

    def w1(self, r):
        return 0

    def w2(self, r):
        return 0

    def w3(self, r):
        return 0

    # The following routines can be used for any wave function ~~~~~~~~~~~~~~~~~

    def Ps(self):
        ''' S-state probability. '''
        def integrand(r):
            return self.u(r)**2
        return quad(integrand, 0, np.inf)[0]

    def Pd(self):
        ''' D-state probability. '''
        def integrand(r):
            return self.w(r)**2
        return quad(integrand, 0, np.inf)[0]

    def radius_squared(self):
        ''' Obtain the mean squared radius associated with the wave function. '''
        def integrand(r):
            return r**2*(self.u(r)**2 + self.w(r)**2)/4
        r2 = quad(integrand, 0, np.inf)[0]
        return r2

    def quadrupole(self):
        ''' Return the quadrupole moment, in fm**2. '''
        def integrand(r):
            return r**2*(2*np.sqrt(2)*self.u(r)*self.w(r) - self.w(r)**2) / 20
        Qd = quad(integrand, 0, np.inf)[0]
        return Qd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Asymptotic forms of the S- and D-waves, and their derivatives

def u_asy(r):
    ''' Asymptotic form of the S-wave for large r. '''
    return np.exp(-kappa*r)

def u1_asy(r):
    ''' Asymptotic form of the first derivative of the S-wave for large r. '''
    return -kappa * np.exp(-kappa*r)

def u2_asy(r):
    ''' Asymptotic form of the second derivative of the S-wave for large r. '''
    return kappa**2 * np.exp(-kappa*r)

def u3_asy(r):
    ''' Asymptotic form of the third derivative of the S-wave for large r. '''
    return -kappa**3 * np.exp(-kappa*r)

def w_asy(r):
    ''' Asymptotic form of the D-wave for large r. '''
    result = np.exp(-kappa*r)*(1 + 3/(kappa*r) + 3/(kappa*r)**2)
    return result

def w1_asy(r):
    ''' Asymptotic form of the first derivative of the D-wave for large r. '''
    result = -kappa * np.exp(-kappa*r)*(
            1 + 3/(kappa*r) + 6/(kappa*r)**2 + 6/(kappa*r)**3
            )
    return result

def w2_asy(r):
    ''' Asymptotic form of the second derivative of the D-wave for large r. '''
    result = kappa**2 * np.exp(-kappa*r)*(
            1 + 3/(kappa*r) + 9/(kappa*r)**2 + 18/(kappa*r)**3 + 18/(kappa*r)**4
            )
    return result

def w3_asy(r):
    ''' Asymptotic form of the third derivative of the D-wave for large r. '''
    result = -kappa**3 * np.exp(-kappa*r)*(
            1 + 3/(kappa*r) + 12/(kappa*r)**2 + 36/(kappa*r)**3
            + 72/(kappa*r)**4 + 72/(kappa*r)**5
            )
    return result

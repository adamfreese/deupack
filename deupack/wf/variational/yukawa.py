# yukawa.py
# budded off from vwf.py on 2026.09.14
#
# Methods to create numerical ground wave functions for Yukawa potentials.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from .vwf import _VARWF
from ...constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A base class for variational wave functions with non-growing potentials.
# Assumes the wave function falls exponentially with distance, and treats the
# decay factor like a fit parameter.
# Should not be used directly; used for defining derived classes.

class _VARWF_EXP(_VARWF):

    def __init__(self,
                 mN = 1, # constituent mass (GeV)
                 N  = 4  # number of terms in the variational approximation
                 ):
        super().__init__(mN=mN, N=N)
        self.bounds[-1] = (1e-4, 11) # do not allow negative decay rate
        # Call the ground state solver in a derived class!
        #self.solve()
        return

    # Parametric wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        u = r*1 # to copy the value instead of identifying the variables
        Nmax = a.shape[0]
        for i in range(Nmax-1):
            u += a[i] * r**(i+2)
        u *= np.exp(-a[-1]*r)
        return u

    def _u1(self, r, a):
        f0 = r*1
        f1 = 1
        Nmax = a.shape[0]
        for i in range(Nmax-1):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
        g0 = np.exp(-a[-1]*r)
        g1 = -a[-1] * g0
        u1 = g0*f1 + g1*f0
        return u1

    def _u2(self, r, a):
        f0 = r*1
        f1 = 1
        f2 = 0
        Nmax = a.shape[0]
        for i in range(Nmax-1):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
            f2 += a[i] * (i+2) * (i+1) * r**i
        g0 = np.exp(-a[-1]*r)
        g1 = -a[-1] * g0
        g2 = a[-1]**2 * g0
        u2 = g0*f2 + 2*g1*f1 + g2*f0
        return u2

    def _u3(self, r, a):
        f0 = r*1
        f1 = 1
        f2 = 0
        f3 = 0
        Nmax = a.shape[0]
        for i in range(Nmax-1):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
            f2 += a[i] * (i+2) * (i+1) * r**i
            if(i > 0):
                f3 += a[i] * (i+2) * (i+1) * i * r**(i-1)
        g0 = np.exp(-a[-1]*r)
        g1 = -a[-1] * g0
        g2 = a[-1]**2 * g0
        g3 = a[-1]**3 * g0
        u2 = g0*f3 + 3*g1*f2 + 3*g2*f1 + g3*f0
        return u2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A base class for variational wave functions with non-growing potentials.
# Assumes the wave function falls exponentially with distance at a known rate.
# Should not be used directly; used for defining derived classes.

class _VARWF_EXP_FIXED(_VARWF):

    def __init__(self,
                 mN = 1, # constituent mass (GeV)
                 N  = 4, # number of terms in the variational approximation
                 k  = 0.1 # exponential decay rate (GeV)
                 ):
        super().__init__(mN=mN, N=N)
        self.kfm = k / hbar
        # Call the ground state solver in a derived class!
        #self.solve()
        return

    # Parametric wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        u = r*1 # to copy the value instead of identifying the variables
        Nmax = a.shape[0]
        for i in range(Nmax):
            u += a[i] * r**(i+2)
        u *= np.exp(-self.kfm*r)
        return u

    def _u1(self, r, a):
        f0 = r*1
        f1 = 1
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
        g0 = np.exp(-self.kfm*r)
        g1 = -self.kfm * g0
        u1 = g0*f1 + g1*f0
        return u1

    def _u2(self, r, a):
        f0 = r*1
        f1 = 1
        f2 = 0
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
            f2 += a[i] * (i+2) * (i+1) * r**i
        g0 = np.exp(-self.kfm*r)
        g1 = -self.kfm * g0
        g2 = self.kfm**2 * g0
        u2 = g0*f2 + 2*g1*f1 + g2*f0
        return u2

    def _u3(self, r, a):
        f0 = r*1
        f1 = 1
        f2 = 0
        f3 = 0
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
            f2 += a[i] * (i+2) * (i+1) * r**i
            if(i > 0):
                f3 += a[i] * (i+2) * (i+1) * i * r**(i-1)
        g0 = np.exp(-self.kfm*r)
        g1 = -self.kfm * g0
        g2 = self.kfm**2 * g0
        g3 = self.kfm**3 * g0
        u2 = g0*f3 + 3*g1*f2 + 3*g2*f1 + g3*f0
        return u2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A derived class of _VARWF_EXP that uses a Yukawa potential. Since the decay
# constant is a fit parameter, it will not generally have the correct asymptotic
# form. However, this is very useful for estimating the ground state energy,
# which in turn can be used to obtain the correct asymptotic form. This class
# is thus used by the true Yukawa solver below.

class _dummy_yukawa(_VARWF_EXP):

    def __init__(self,
                 mN    = 1,   # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mu    = 0.1, # screening parameters
                 alpha = 1    # interaction strength
                 ):
        self.mu = mu
        self.alpha = alpha
        super().__init__(mN=mN, N=N)
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return -self.alpha * np.exp(-self.mu*r/hbar)/r

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A class to solve for the Yukawa potential's ground state using variational
# methods. First obtains an estimate of the ground state energy, and then
# estimates the wave function with the correct asymptotic form.

class vwf_yukawa(_VARWF_EXP_FIXED):

    def __init__(self,
                 mN    = 1,   # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mu    = 0.1, # screening parameters
                 alpha = 1    # interaction strength
                 ):
        self.mu = mu
        self.alpha = alpha
        # First, allow decay parameter to float. Get the ground state energy.
        _dummy = _dummy_yukawa(mN=mN, N=N+1, mu=mu, alpha=alpha)
        k = np.sqrt(-mN*_dummy.E)
        # Set up a fixed-decay form
        super().__init__(mN=mN, N=N, k=k)
        # Now run the solver with the correct asymptotic form
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return -self.alpha * np.exp(-self.mu*r/hbar)/r

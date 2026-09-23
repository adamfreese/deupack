# asymptotic.py
# budded off from vwf.py on 2026.09.14
#
# Methods to create numerical ground wave functions for potentials that
# asymptotically grow as some power of r.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from .vwf import _VARWF
from ...constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A base class for variational wave functions with growing potentials.
# The potential should grow like some power
#   V(r) ~ r**n
# at large r.
# Should not be used directly; used for defining derived classes.

class _VARWF_ASY(_VARWF):

    def __init__(self,
                 mN    = 1.4, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 rbig  = 1e5, # estimate of when r is big enough for asymptotic
                              # form to take over
                 name  = ''
                 ):
        super().__init__(mN=mN, N=N, name=name)
        # Internal parameters
        self.n_asy = (
                np.log( self._Vfun(rbig*2)
                       /self._Vfun(rbig) )
                / np.log(2)
                )
        self.Vnfm = self._Vfun(rbig) / rbig**self.n_asy
        # Call the ground state solver in a derived class!
        #self.solve()
        return

    # Parametric wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mredfm*self.Vnfm)/(self.n_asy+2)
        u = r*1 # to copy the value instead of identifying the variables
        Nmax = a.shape[0]
        for i in range(Nmax):
            u += a[i] * r**(i+2)
        u *= np.exp(-L*r**(n/2+1))
        return u

    def _u1(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mredfm*self.Vnfm)/(self.n_asy+2)
        f0 = r*1
        f1 = 1
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
        g0 = np.exp(-L*r**(n/2+1))
        g1 = -(n/2+1)*L*r**(n/2) * g0
        u1 = g0*f1 + g1*f0
        return u1

    def _u2(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mredfm*self.Vnfm)/(self.n_asy+2)
        f0 = r*1
        f1 = 1
        f2 = 0
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
            f2 += a[i] * (i+2) * (i+1) * r**i
        g0 = np.exp(-L*r**(n/2+1))
        g1 = -(n/2+1)*L*r**(n/2) * g0
        g2 = 1/4*((2+n)**2*L**2*r**n - n*(2+n)*L*r**(n/2-1)) * g0
        u2 = g0*f2 + 2*g1*f1 + g2*f0
        return u2

    def _u3(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mredfm*self.Vnfm)/(self.n_asy+2)
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
        g0 = np.exp(-L*r**(n/2+1))
        g1 = -(n/2+1)*L*r**(n/2) * g0
        g2 = 1/4*((2+n)**2*L**2*r**n - n*(2+n)*L*r**(n/2-1)) * g0
        g3 = 1/8*(
                - (2+n)**3*L**3*r**(3*n/2)
                + 2*n*(2+n)**2*L**2*r**(n-1)
                - (n-1)*n*(n+1)*L*r**(n/2-2)
                ) * g0
        u2 = g0*f3 + 3*g1*f2 + 3*g2*f1 + g3*f0
        return u2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Class that solves for the ground state of the Cornell potential

class vwf_cornell(_VARWF_ASY):

    def __init__(self,
                 mN    = 1.4, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 sigma = 0.136, # QCD string tension (GeV**2)
                 alpha = 0.472, # 4/3 * alphaQCD at dressed charm mass
                 name  = ''
                 ):
        # Internal parameters
        self.sigma = sigma
        self.alpha = alpha
        # Base class initialization
        super().__init__(mN=mN, N=N, name=name)
        # Override _VARWF_ASY parameters with exact values
        self.n_asy = 1
        self.Vnfm = sigma / hbar**2
        # Call the ground state solver
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return self.sigma/hbar**2*r - self.alpha/r

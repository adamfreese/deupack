# hydrogen.py
# Created 2026.08.13 by Adam Freese
#
# A class for hydrogen atom wave functions.

import numpy as np
from scipy.special import assoc_laguerre, gamma

from .dwf import DWF
from ..constants import hbar, alphaQED

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dwf_hydrogen(DWF):
    ''' Creates a wave function for a non-relativistic hydrogen-like atom
    and hijacks the machinery of deupack to calculate its EMT-FFs.

    Because of numerical instability, n>159 does not work.
    '''

    def __init__(self,
                 n = 1,
                 l = 0,
                 ml = 0,
                 mN = 0.106,   # muon mass (GeV)
                 alpha = alphaQED
                 ):
        if(l >= n):
            raise ValueError(
                    "n={:d} and l={:d} invalid; you must give l < n".format(n, l)
                    )
        if(ml > l or ml < -l):
            raise ValueError(
                    "l={:d} and ml={:d} invalid; you must give -l <= ml <= l".format(l, ml)
                    )
        super().__init__()
        self.name = 'hydrogen_{:d}_{:d}_{:d}'.format(n,l,ml)
        self.mN = mN
        self.mNfm = mN / hbar
        self.mred = mN / 2
        self.alpha = alpha
        # Quantum numbers
        self.n = n
        self.l = l
        self.ml = ml
        # Energy
        self.E = -self.mred * self.alpha**2/(2*n**2)
        # Handy thingies
        self.a0 = 2*hbar/(alpha*mN)
        self.kappa = 1/(self.n*self.a0)
        self.N = np.sqrt( self.kappa * gamma(n-l) / n / gamma(n+l+1) )
        self.rmax = 17 * n**2 * self.a0 # not sure why this works
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        X = 2*self.kappa*r
        L = assoc_laguerre(X, self.n-self.l-1, 2*self.l+1)
        return self.N * X**(self.l+1) * np.exp(-X/2) * L

    def u1(self, r):
        X = 2*self.kappa*r
        L1 = assoc_laguerre(X, self.n-self.l-1, 2*self.l+1)
        L2 = assoc_laguerre(X, self.n-self.l-2, 2*self.l+2)
        terms = (self.l + 1 - X/2) * L1
        if(self.n >= 2+self.l):
            terms -= X * L2
        return 2*self.kappa*self.N * X**self.l * np.exp(-X/2) * terms

    def u2(self, r):
        factor = 2 * self.mred/hbar * (-self.alpha/r - self.E/hbar)
        return factor * self.u(r)

    def u3(self, r):
        factor1 = 2 * self.mred/hbar * (-self.alpha/r - self.E/hbar)
        factor2 = 2 * self.mred/hbar * self.alpha/r**2
        return factor1*self.u1(r) + factor2*self.u(r)

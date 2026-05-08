# airy.py
# Created 2026.05.08 by Adam Freese
#
# A class for Airy wave functions, as solutions to a linear potential.

import numpy as np
from scipy.special import airy, ai_zeros

from .dwf import DWF
from ..constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_N = 31
_zeros, *_ = ai_zeros(_N)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dwf_airy(DWF):
    ''' Creates a wave function that is a solution to the potential
        V(r) = sigma*r
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    '''

    def __init__(self,
                 n = 0,
                 mN = 1.4,     # dressed charm mass (GeV)
                 sigma = 0.136 # QCD string tension (GeV**2)
                 ):
        super().__init__()
        self.name = 'airy{:d}'.format(n)
        self.mN = mN
        self.mNfm = mN / hbar
        self.sigma = sigma / hbar**2
        # Index of the Airy function offset and the associated zero
        self.n = n # ground state
        self.an = _zeros[self.n]
        # Normalization factor and energy
        _, Aip_N, *_ = airy(self.an)
        self.N = (self.mNfm*self.sigma)**(1/6) / Aip_N
        self.Efm = -self.sigma*self.an/(self.mNfm*self.sigma)**(1/3) # fm**-1
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        x = (self.mNfm*self.sigma)**(1/3)*r + self.an
        Ai, *_ = airy(x)
        #_, Aip_N, *_ = airy(self.an)
        return self.N*Ai

    def u1(self, r):
        x = (self.mNfm*self.sigma)**(1/3)*r + self.an
        _, Aip, *_ = airy(x)
        return self.N*Aip * (self.mNfm*self.sigma)**(1/3) # chain rule

    def u2(self, r):
        #return (self.mNfm*self.sigma*r+self.an)*self.u(r)
        return self.mNfm*self.sigma*(r - self.Efm/self.sigma) * self.u(r)

    def u3(self, r):
        piece1 = self.mNfm*self.sigma*(r - self.Efm/self.sigma) * self.u1(r)
        piece2 = self.mNfm*self.sigma                           * self.u(r)
        return piece1 + piece2

    # Other internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def energy(self):
        return -self.sigma*self.an/(self.mNfm*self.sigma)**(1/3)

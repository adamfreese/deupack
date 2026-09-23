# multifield.py
# created 2026.09.16 by Adam Freese
#
# Find a ground state wave function for a potential with an attractive Yukawa
# field, as well as electrostatic and gravitational fields.
# static interactions.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from .yukawa import _VARWF_EXP, _VARWF_EXP_FIXED
from ...constants import hbar, alphaQED, GN

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _dummy_multifield(_VARWF_EXP):

    def __init__(self,
                 mN    = 1,   # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mu    = 0.1, # screening parameter
                 alpha = 1,   # Yukawa interaction strength
                 sign  = 1,   # relative sign of charges
                 name  = ''
                 ):
        self.mu = mu
        self.alpha = alpha
        self.sign = sign
        super().__init__(mN=mN, N=N, name=name)
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        term1 =-self.alpha * np.exp(-self.mu*r/hbar) / r
        term2 = self.sign * alphaQED / r
        term3 = -GN*self.mN**2 / r
        return term1 + term2 + term3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A class to solve for the Yukawa potential's ground state using variational
# methods. First obtains an estimate of the ground state energy, and then
# estimates the wave function with the correct asymptotic form.

class vwf_multifield(_VARWF_EXP_FIXED):

    def __init__(self,
                 mN    = 1,   # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mu    = 0.1, # screening parameter
                 alpha = 1,   # Yukawa interaction strength
                 sign  = 1,   # relative sign of charges
                 name  = ''
                 ):
        self.mu = mu
        self.alpha = alpha
        self.sign = sign
        # First, allow decay parameter to float. Get the ground state energy.
        _dummy = _dummy_multifield(mN=mN, N=N+1, mu=mu, alpha=alpha, sign=sign, name=name)
        k = np.sqrt(-mN*_dummy.E)
        # Set up a fixed-decay form
        super().__init__(mN=mN, N=N, k=k)
        # Now run the solver with the correct asymptotic form
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        term1 =-self.alpha * np.exp(-self.mu*r/hbar) / r
        term2 = self.sign * alphaQED / r
        term3 = -GN*self.mN**2 / r
        return term1 + term2 + term3

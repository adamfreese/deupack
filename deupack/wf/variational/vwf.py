# vwf.py
# Created 2026.08.14 by Adam Freese
#
# Methods to create numerical ground wave functions for various potentials.
# In progress.
# TODO:
# - coupled channels (e.g., S and D waves)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from ..dwf import DWF
from ...constants import hbar, mN

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A base class for variational wave functions.
# Should not be used directly; used for defining derived classes.

class _VARWF(DWF):
    ''' Creates a wave function that is an approximate ground state solution
    to the potential given upon initialization. This is a base class that needs
    to have _u, _u1, _u2 and _u3 overridden by a particular parametric form.
    '''
    def __init__(self,
                 mN   = 1, # constituent mass (GeV)
                 N    = 4, # number of parameters to use (integer, >0)
                 name = 'variational'
                 ):
        super().__init__()
        # Internal parameters
        self.name = name
        self.mN = mN
        self.mNfm = mN / hbar
        # Properties related to the ground state solver
        self.N = N
        self.bounds = [ (-1, 1) for _ in range(self.N) ]
        self.mredfm = mN / hbar / 2
        # Call the ground state solver in the derived class!
        #self.solve()
        return

    def solve(self):
        stuff = differential_evolution(
            _energy,
            self.bounds,
            args = (self,),
            popsize = 32,
            workers = 8,
            updating = 'deferred',
            tol = 1e-5,
            maxiter = 5000
            )
        self.a = stuff['x']
        N2 = quad(_usq_integrand, 0, np.inf,
                  args = (self.a, self)
                  )[0]
        self.C = 1/np.sqrt(N2)
        self.Efm = stuff['fun']
        self.E = self.Efm * hbar
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        return self._u(r, self.a) * self.C

    def u1(self, r):
        return self._u1(r, self.a) * self.C

    def u2(self, r):
        return self._u2(r, self.a) * self.C

    def u3(self, r):
        return self._u3(r, self.a) * self.C

    # Parametric wave function (used by derived classes) ~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        return 0

    def _u1(self, r, a):
        return 0

    def _u2(self, r, a):
        return 0

    def _u3(self, r, a):
        return 0

    # Potential energy function (used by derived classes) ~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Auxiliary methods used by the variational solver

def _energy(a, wf):
    ''' Expectation value of energy for a variational state.
    Input:
    - a ....... numpy array with coefficients in the variational wave function
    - wf ...... VARWF object
    See solve_potential for further details
    '''
    num = quad(_energy_integrand, 0, np.inf,
               args = (a, wf),
               )[0]
    den = quad(_usq_integrand, 0, np.inf,
               args = (a, wf),
               )[0]
    return num/den

def _energy_integrand(r, a, wf):
    '''' Integrand for expected value of energy. '''
    u  = wf._u(r, a)
    u2 = wf._u2(r, a)
    V = wf._Vfun(r)
    return u * ( V*u - u2/(2*wf.mredfm))

def _usq_integrand(r, a, wf):
    ''' u**2(r) --- to find normalization. '''
    u = wf._u(r, a)
    return u**2

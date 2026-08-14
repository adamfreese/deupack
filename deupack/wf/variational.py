# variational.py
# Created 2026.08.14 by Adam Freese
#
# Methods to create numerical wave functions that approximate
# ground states of various potentials, whose large distance
# behavior is dominated by a term r**n for n > 0.
# In progress.
# TODO
# - Better names for variables (to make code easier to read)
# - Arbitrary potential (instead of polynomial), but with growth in r
# - docstrings
# NOTE: this module is unstable! I will change how potentials are dealt with.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from .dwf import DWF
from ..constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some wave functions using particular potentials

class var_wf(DWF):
    ''' Creates a wave function that is an approximate solution to the potential
        V(r) = Vcoeff[0]*r**nmin + Vcoeff[1]*r**(nmin+1) + ...
    and hijacks the machinery of deupack to calculate its EMT-FFs.

    Uses a variational method to approximate the ground state.
    Assumes a form of:
        u(r) = C*r*exp(-2*sqrt(2*mu*Vn)/(n+2)*r**(n/2+1))*(1 + a[0]*r + a[1]*r**2 + ...)
    and solves for a that minimizes the energy.
    This method only works if the highest power of r in V(r) is at least 1.
    '''
    def __init__(self,
                 mN   = 1, # constituent mass (GeV)
                 N    = 4, # number of terms in the variational approximation
                 nmin = 1, # lowest power of r in V(r)
                 Vcoeff = np.array([1]) # coefficients of powers of r in V(r)
                 ):
        super().__init__()
        # Internal parameters
        self.name = 'variational'
        self.mN = mN
        self.mNfm = mN / hbar
        # Properties related to the ground state solver
        self.N = N
        self.mu = mN / hbar / 2
        self.nmin = nmin
        self.Vcoeff = Vcoeff
        self.n = self.nmin + self.Vcoeff.shape[0] - 1
        self.Vn = self.Vcoeff[-1]
        self.L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n+2)
        # Call the ground state solver
        self._solve()
        return

    def _solve(self):
        a, C, E = solve_potential(self.nmin, self.Vcoeff, self.mu, N=self.N)
        self.a = a
        self.C = C
        self.E = E*hbar
        self.Efm = E
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        return _u(r, self.n, self.L, self.a) * self.C

    def u1(self, r):
        return _u1(r, self.n, self.L, self.a) * self.C

    def u2(self, r):
        return _u2(r, self.n, self.L, self.a) * self.C

    def u3(self, r):
        # TODO
        return 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class var_wf_airy(var_wf):
    ''' Creates a wave function that is an approximate solution to the potential
        V(r) = sigma*r
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    Uses a variational method to approximate the ground state.
    '''

    def __init__(self,
                 mN = 1.4,      # dressed charm mass (GeV)
                 sigma = 0.136, # QCD string tension (GeV**2)
                 N = 4          # number of terms in the variational approximation
                 ):
        super().__init__(mN=mN,
                         N=N,
                         nmin=1,
                         Vcoeff = np.array([sigma/hbar**2])
                         )
        # Internal parameters
        self.name = 'var_airy_N'.format(N)
        self.sigma = sigma / hbar**2
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class var_wf_harmonic(var_wf):
    ''' Creates a wave function that is an approximate solution to the potential
        V(r) = 1/2*mu*omega0*r**2
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    Uses a variational method to approximate the ground state.
    '''

    def __init__(self,
                 mN = 1.4,      # dressed charm mass (GeV)
                 omega0 = 1,    # TODO: reasonable default
                 N = 4          # number of terms in the variational approximation
                 ):
        mu = mN/hbar/2
        k = mu*omega0**2
        super().__init__(mN=mN,
                         N=N,
                         nmin=2,
                         Vcoeff = np.array([k/2])
                         )
        # Internal parameters
        self.name = 'var_harmonic_N'.format(N)
        self.k = k # TODO
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class var_wf_cornell(var_wf):
    ''' Creates a wave function that is an approximate solution to the potential
        V(r) = sigma*r - alpha/r
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    Uses a variational method to approximate the ground state.
    In this context, alpha=alphaQED for an electromagnetic interaction,
    and alpha = CF*alphaQCD for the strong interaction.
    '''

    def __init__(self,
                 mN = 1.4,      # dressed charm mass (GeV)
                 sigma = 0.136, # QCD string tension (GeV**2)
                 alpha = 0.472, # 4/3 * alphaQCD at dressed charm mass
                 N = 4          # number of terms in the variational approximation
                 ):
        super().__init__(mN=mN,
                         N=N,
                         nmin=-1,
                         Vcoeff = np.array([-alpha, 0, sigma/hbar**2])
                         )
        # Internal parameters
        self.name = 'var_cornell_N'.format(N)
        self.sigma = sigma / hbar**2
        self.alpha = alpha
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Potential solver routine

def solve_potential(nmin, Vcoeff, mu, N=4):
    '''
    Considers a potential
        V(r) = Vcoeff[0]*r**nmin + Vcoeff[1]*r**(nmin+1) + ...
    binding two particles with a reduced mass mu and approximately solves for
    the wave function, using an ansatz
        u(r) = C*r*exp(-2*sqrt(2*mu*Vn)/(n+2)*r**(n/2+1))*(1 + a[0]*r + a[1]*r**2 + ...)
    This method returns (a,E).
    Note that this method assumes a maximum power > 0.
    ------
    Input:
    - nmin ..... smallest power of r appearing in the potential
    - Vcoeff ... coefficients of powers of r appearing in the potential
                 each number should be in units of a power of fm
    - mu ....... reduced mass (fm**-1)
    ------
    Optional input:
    - N ........ number of a coefficients (default N=4)
    ------
    Output:
    - a ........ coefficients of the r powers in u(r)
                 each number will be in units of a power of fm
    - C ........ factor to multiply wave function by to normalize it
    - E ........ ground state energy (fm**-1)
    '''
    bounds = [ (-7, 7) for _ in range(N) ]
    stuff = differential_evolution(
            _energy,
            bounds,
            args = (nmin,Vcoeff,mu),
            popsize = 32,
            workers = 8,
            tol = 0.0001,
            maxiter = 2600
            )
    a = stuff['x']
    N2 = quad(_usq_integrand, 0, np.inf,
              args = (nmin, Vcoeff, a, mu)
              )[0]
    C = 1/np.sqrt(N2)
    E = stuff['fun']
    return a, C, E

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper routines for the potential solver

def _energy(a, nmin, Vcoeff, mu):
    # TODO: docstring
    num = quad(_energy_integrand, 0, np.inf,
               args = (nmin, Vcoeff, a, mu)
               )[0]
    den = quad(_usq_integrand, 0, np.inf,
               args = (nmin, Vcoeff, a, mu)
               )[0]
    return num/den

def _energy_integrand(r, nmin, Vcoeff, a, mu):
    # TODO: docstring
    n = nmin + Vcoeff.shape[0] - 1
    Vn = Vcoeff[-1]
    L = 2*np.sqrt(2*mu*Vn)/(n+2)
    u = _u(r, n, L, a)
    u2 = _u2(r, n, L, a)
    V = _V(r, nmin, Vcoeff)
    return u * ( V*u - u2/(2*mu))

def _usq_integrand(r, nmin, Vcoeff, a, mu):
    # TODO: docstring
    n = nmin + Vcoeff.shape[0] - 1
    Vn = Vcoeff[-1]
    L = 2*np.sqrt(2*mu*Vn)/(n+2)
    u = _u(r, n, L, a)
    return u**2

def _V(r, nmin, Vcoeff):
    # TODO: docstring
    V = 0
    for i in range(Vcoeff.shape[0]):
        V += Vcoeff[i] * r**(i + nmin)
    return V

def _u(r, n, L, a):
    ''' Approximate form of the u(r) wave function.
        u(r) = C*r*exp(-2*sqrt(2*mu*Vn)/(n+2)*r**(n/2+1))*(1 + a[0]*r + a[1]*r**2 + ...)
    Input:
        - r .... float or array of floats; separation (fm)
        - n .... float; highest power of r in potential
        - L .... float, = 2*(sqrt(2*mu*Vn)/(n+2)) (in fm**(-(n/2+1)))
        - a .... array of floats; a[i-1] is coefficient of r**i
    Output:
        Float or array of floats with shape of r (fm**1/2)
    '''
    u = r*1 # to copy the value instead of identifying the variables
    Nmax = a.shape[0]
    for i in range(Nmax):
        u += a[i] * r**(i+2)
    u *= np.exp(-L*r**(n/2+1))
    return u

def _u1(r, n, L, a):
    ''' First derivative of _u. See docstring thereof for details. '''
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

def _u2(r, n, L, a):
    ''' Second derivative of _u. See docstring thereof for details. '''
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

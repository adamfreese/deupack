# variational.py
# Created 2026.08.14 by Adam Freese
#
# Methods to create numerical wave functions that approximate
# ground states of various potentials, whose large distance
# behavior is dominated by a term r**n for n > 0.
# In progress.
# NOTE: this module is unstable! its interface is expected to change
# TODO
# - Better names for variables (to make code easier to read)
# - Deal with non-growing potentials
# - docstrings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from .dwf import DWF
from ..constants import hbar

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Python's mutliprocessing capabilities cannot deal with lambas and nested
# functions. To mitigate this, we use a Potential class that can keep a
# potential function as a member. A Potential class object is passed to
# the variational wave function class instead of a function directly.

class Potential:
    ''' ad hoc workaround for passing parametrized functions into multithreaded
    routines.
    '''
    def __init__(self):
        return
    def Vfun(self, r):
        return 0

class MonomialPotential(Potential):
    ''' Parametrizes a potential of the form
        V(r) = Vn*r**n
    where Vn has units fm**(-1-n).
    '''
    def __init__(self, n, Vn):
        self.n = n
        self.Vn = Vn
        return
    def Vfun(self, r):
        return self.Vn * r**self.n


class CornellPotential(Potential):
    def __init__(self, alpha, sigma):
        ''' Parametrizes a potential of the form
            V(r) = sigma*r - alpha/r
        where sigma is in fm**-2 and alpha is unitless.
        Note that alpha here really means alphaQCD*CF.
        '''
        self.sigma = sigma
        self.alpha = alpha
        return
    def Vfun(self, r):
        return self.sigma*r - self.alpha/r

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some wave functions using particular potentials

class var_wf(DWF):
    ''' Creates a wave function that is an approximate solution to the potential
        V(r) = Vfun(r)
    with an asymptotic form
        V(r) ~ Vn*r**n_asy
    at large r. Solves for the ground state wave function using an ansatz
        u(r) = C*r*exp(-2*sqrt(2*mu*Vn)/(n+2)*r**(n/2+1))*(1 + a[0]*r + a[1]*r**2 + ...)
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    '''
    def __init__(self,
                 mN    = 1, # constituent mass (GeV)
                 N     = 4, # number of terms in the variational approximation
                 n_asy = 1, # asymptotic power of V(r)
                 Vn    = 1, # coefficient of asymptotic power
                 potential = None
                 ):
        super().__init__()
        # Internal parameters
        self.name = 'variational'
        self.mN = mN
        self.mNfm = mN / hbar
        # Properties related to the ground state solver
        self.N = N
        self.mu = mN / hbar / 2
        self.n_asy = n_asy
        self.Vn = Vn
        self.L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n_asy+2)
        # If user does not supply a Vfun, use a monomial
        if(potential is None):
            self.potential = MonomialPotential(n=n_asy, Vn=Vn)
        else:
            self.potential = potential
        # Call the ground state solver
        self._solve()
        return

    def _solve(self):
        a, C, E = solve_potential(self.mu, self.n_asy, self.Vn, self.potential, N=self.N)
        self.a = a
        self.C = C
        self.E = E*hbar
        self.Efm = E
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        return _u(r, self.n_asy, self.L, self.a) * self.C

    def u1(self, r):
        return _u1(r, self.n_asy, self.L, self.a) * self.C

    def u2(self, r):
        return _u2(r, self.n_asy, self.L, self.a) * self.C

    def u3(self, r):
        return _u3(r, self.n_asy, self.L, self.a) * self.C

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
        super().__init__(mN=mN, N=N, n_asy=1, Vn=sigma/hbar**2)
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
        super().__init__(mN=mN, N=N, n_asy=2, Vn=k/2)
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
        sigma_fm = sigma/hbar**2
        # Cornell potential function
        def _Vc(r):
            return sigma_fm*r - alpha/r
        super().__init__(
                mN=mN, N=N, n_asy=1, Vn=sigma_fm,
                potential=CornellPotential(sigma=sigma_fm, alpha=alpha)
                )
        # Internal parameters
        self.name = 'var_cornell_N'.format(N)
        self.sigma = sigma_fm
        self.alpha = alpha
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Potential solver routine

def solve_potential(mu, n_asy, Vn, pot, N=4):
    '''
    Considers a potential
        V(r) = pot.Vfun(r)
    binding two particles with a reduced mass mu, which has the asymptotic form
        V(r) ~ Vn*r**n_asy
    at large r. Solves for the ground state wave function using an ansatz
        u(r) = C*r*exp(-2*sqrt(2*mu*Vn)/(n+2)*r**(n/2+1))*(1 + a[0]*r + a[1]*r**2 + ...)
    This method returns (a,C,E).
    The method requires n_asy > 0 and Vn > 0 to obtain a sensible result.
    ------
    Input:
    - mu ....... float
                 reduced mass (fm**-1)
    - n_asy .... integer or float
                 asymptotic power of Vfun(r)
                 must be > 0
    - Vn ....... float
                 coefficient of r**n_asy (fm**(-1-n_asy))
                 must be > 0
    - pot ...... Potential object
                 contains Vfun as a member
                 Vfun is a float function (takes float);
                 potential energy function, as a function of r (fm**-1)
    ------
    Optional input:
    - N ........ integer
                 number of a coefficients (default N=4)
    ------
    Output:
    - a ........ numpy.array of floats
                 coefficients of the r powers in u(r)
                 each number will be in units of a power of fm
    - C ........ float
                 factor to multiply wave function by to normalize it
    - E ........ float
                 ground state energy (fm**-1)
    '''
    bounds = [ (-7, 7) for _ in range(N) ]
    stuff = differential_evolution(
            _energy,
            bounds,
            args = (mu, n_asy, Vn, pot),
            popsize = 32,
            workers = 8,
            tol = 0.0001,
            maxiter = 2600
            )
    a = stuff['x']
    N2 = quad(_usq_integrand, 0, np.inf,
              args = (a, mu, n_asy, Vn)
              )[0]
    C = 1/np.sqrt(N2)
    E = stuff['fun']
    return a, C, E

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper routines for the potential solver

def _energy(a, mu, n_asy, Vn, pot):
    ''' Expectation value of energy for a variational state.
    Input:
    - a ....... numpy array with coefficients in the variational wave function
    - mu ...... reduced mass in fm
    - n_asy ... asymptotic power of V(r)
    - Vn ...... coefficient of asymptotic power of V(r)
    - pot ..... Potential object containing Vfun
    See solve_potential for further details
    '''
    num = quad(_energy_integrand, 0, np.inf,
               args = (a, mu, n_asy, Vn, pot),
               )[0]
    den = quad(_usq_integrand, 0, np.inf,
               args = (a, mu, n_asy, Vn),
               )[0]
    return num/den

def _energy_integrand(r, a, mu, n_asy, Vn, pot):
    '''' Integrand for expected value of energy. '''
    L = 2*np.sqrt(2*mu*Vn)/(n_asy+2)
    u = _u(r, n_asy, L, a)
    u2 = _u2(r, n_asy, L, a)
    V = pot.Vfun(r)
    return u * ( V*u - u2/(2*mu))

def _usq_integrand(r, a, mu, n_asy, Vn):
    ''' u**2(r) --- to find normalization. '''
    L = 2*np.sqrt(2*mu*Vn)/(n_asy+2)
    u = _u(r, n_asy, L, a)
    return u**2

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

def _u3(r, n, L, a):
    ''' Third derivative of _u. See docstring thereof for details. '''
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

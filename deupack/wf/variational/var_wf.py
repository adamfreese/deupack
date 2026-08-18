# var_wf.py
# Created 2026.08.14 by Adam Freese
#
# Methods to create numerical ground wave functions for various potentials.
# In progress.
# NOTE: this module is unstable! its interface is expected to change

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from ..dwf import DWF
from ...constants import hbar, mpi_0, mN, a0

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
        self.bounds = [ (-11, 11) for _ in range(self.N) ] # TODO: generalize
        self.mu = mN / hbar / 2
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
# A base class for variational wave functions with growing potentials.
# The potential should grow like some power
#   V(r) ~ r**n
# at large r.
# Should not be used directly; used for defining derived classes.

class _VARWF_ASY(_VARWF):

    def __init__(self,
                 mN    = 1.4, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 rbig  = 1e5  # estimate of when r is big enough for asymptotic
                              # form to take over
                 ):
        super().__init__(mN=mN, N=N)
        # Internal parameters
        self.n_asy = (
                np.log( self._Vfun(rbig*2)
                       /self._Vfun(rbig) )
                / np.log(2)
                )
        self.Vn = self._Vfun(rbig) / rbig**self.n_asy
        # Call the ground state solver in a derived class!
        #self.solve()
        return

    # Parametric wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n_asy+2)
        u = r*1 # to copy the value instead of identifying the variables
        Nmax = a.shape[0]
        for i in range(Nmax):
            u += a[i] * r**(i+2)
        u *= np.exp(-L*r**(n/2+1))
        return u

    def _u1(self, r, a):
        n = self.n_asy
        L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n_asy+2)
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
        L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n_asy+2)
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
        L = 2*np.sqrt(2*self.mu*self.Vn)/(self.n_asy+2)
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
# A base class for variational wave functions with non-growing potentials.
# Assumes the wave function falls exponentially with distance, and treats the
# decay factor like a fit parameter.
# Should not be used directly; used for defining derived classes.

class _VARWF_EXP(_VARWF):

    def __init__(self,
                 mN    = 1.4, # constituent mass (GeV)
                 N     = 4    # number of terms in the variational approximation
                 ):
        super().__init__(mN=mN, N=N)
        self.bounds[-1] = (1/(2*a0), 11) # do not allow negative a[-1]
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
# A derived class of _VARWF_EXP that uses a Yukawa potential. Since the decay
# constant is a fit parameter, it will not generally have the correct asymptotic
# form. However, this is very useful for estimating the ground state energy,
# which in turn can be used to obtain the correct asymptotic form. This class
# is thus used by the true Yukawa solver below.

class _dummy_yukawa(_VARWF_EXP):

    def __init__(self,
                 mN    = mN, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mY    = mpi_0/hbar, # meson exchange mass
                 alpha = 1 # TODO: sane default
                 ):
        self.mY = mY
        self.alpha = alpha
        super().__init__(mN=mN, N=N)
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return -self.alpha * np.exp(-self.mY*r)/r

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A class to solve for the Yukawa potential's ground state using variational
# methods. First obtains an estimate of the ground state energy, and then
# estimates the wave function with the correct asymptotic form.

class vwf_yukawa(_VARWF):

    def __init__(self,
                 mN    = mN, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 mY    = mpi_0/hbar, # meson exchange mass
                 alpha = 1 # TODO: sane default
                 ):
        self.mY = mY
        self.alpha = alpha
        super().__init__(mN=mN, N=N)
        # First, get a good estimate of the ground state energy from the
        # _dummy_yukawa object with N+2 terms. This energy gives the correct
        # asymptotic form of the Yukawa wave function, so we shouldn't let
        # the coefficient in the exponential float.
        _dummy = _dummy_yukawa(mN=mN, N=N+2, mY=mY, alpha=alpha)
        Efm = _dummy.Efm
        self.kappa = np.sqrt(-2*self.mu*Efm)
        # Now we solve for real
        self.solve()
        return

    # Parametric wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _u(self, r, a):
        u = r*1 # to copy the value instead of identifying the variables
        Nmax = a.shape[0]
        for i in range(Nmax):
            u += a[i] * r**(i+2)
        u *= np.exp(-self.kappa*r)
        return u

    def _u1(self, r, a):
        f0 = r*1
        f1 = 1
        Nmax = a.shape[0]
        for i in range(Nmax):
            f0 += a[i] * r**(i+2)
            f1 += a[i] * (i+2) * r**(i+1)
        g0 = np.exp(-self.kappa*r)
        g1 = -self.kappa * g0
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
        g0 = np.exp(-self.kappa*r)
        g1 = -self.kappa * g0
        g2 = self.kappa**2 * g0
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
        g0 = np.exp(-self.kappa*r)
        g1 = -self.kappa * g0
        g2 = self.kappa**2 * g0
        g3 = self.kappa**3 * g0
        u2 = g0*f3 + 3*g1*f2 + 3*g2*f1 + g3*f0
        return u2

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return -self.alpha * np.exp(-self.mY*r)/r

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Class that solves for the ground state of the Cornell potential

class vwf_cornell(_VARWF_ASY):

    def __init__(self,
                 mN    = 1.4, # constituent mass (GeV)
                 N     = 4,   # number of terms in the variational approximation
                 sigma = 0.136, # QCD string tension (GeV**2)
                 alpha = 0.472, # 4/3 * alphaQCD at dressed charm mass
                 ):
        # Internal parameters
        self.sigma = sigma/hbar**2
        self.alpha = alpha
        # Base class initialization
        super().__init__(mN=mN, N=N)
        # Override _VARWF_ASY parameters with exact values
        self.n_asy = 1
        self.Vn = sigma / hbar**2
        # Call the ground state solver
        self.solve()
        return

    # Potential energy override ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _Vfun(self, r):
        return self.sigma*r - self.alpha/r

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
    return u * ( V*u - u2/(2*wf.mu))

def _usq_integrand(r, a, wf):
    ''' u**2(r) --- to find normalization. '''
    u = wf._u(r, a)
    return u**2

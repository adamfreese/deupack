# var_wf.py
# Created 2026.08.14 by Adam Freese
#
# Methods to create numerical ground wave functions for various potentials.
# In progress.
# NOTE: this module is unstable! its interface is expected to change
# TODO
# - Better names for variables (to make code easier to read)
# - Better way to deal with Yukawa potential
# - docstrings

import numpy as np

from . import growing
from . import shrinking

from ..dwf import DWF
from ...constants import hbar, alphaQED, mpi_0

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

class CoulombPotential(Potential):
    def __init__(self, alpha=alphaQED):
        ''' Parametrizes a potential of the form
            V(r) = - alpha/r
        '''
        self.alpha = alpha
        return
    def Vfun(self, r):
        return -self.alpha/r

class YukawaPotential(Potential):
    def __init__(self, alpha=1, mu=mpi_0/hbar):
        ''' Parametrizes a potential of the form
            V(r) = - alpha/r * exp(-mu*r)
        where mu is in fm**-1.
        '''
        # TODO: sensible default alpha
        self.alpha = alpha
        self.mu = mu
        return
    def Vfun(self, r):
        return -self.alpha/r*np.exp(-self.mu*r)

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
        if(potential is None and n_asy > 0):
            self.potential = MonomialPotential(n=n_asy, Vn=Vn)
        elif(potential is None):
            # TODO: sensible shrinking potential
            self.potential = CoulombPotential()
        else:
            self.potential = potential
        # Call the ground state solver
        self._solve()
        return

    def _solve(self):
        if(self.n_asy > 0):
            a, C, E = growing.solve_potential(self.mu, self.n_asy, self.Vn, self.potential, N=self.N)
        else:
            a, C, E = shrinking.solve_potential(self.mu, self.potential, N=self.N)
        self.a = a
        self.C = C
        self.E = E*hbar
        self.Efm = E
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        if(self.n_asy > 0):
            return growing.u_func(r, self.n_asy, self.L, self.a) * self.C
        else:
            return shrinking.u_func(r, self.a) * self.C

    def u1(self, r):
        if(self.n_asy > 0):
            return growing.u1_func(r, self.n_asy, self.L, self.a) * self.C
        else:
            return shrinking.u1_func(r, self.a) * self.C

    def u2(self, r):
        if(self.n_asy > 0):
            return growing.u2_func(r, self.n_asy, self.L, self.a) * self.C
        else:
            return shrinking.u2_func(r, self.a) * self.C

    def u3(self, r):
        if(self.n_asy > 0):
            return growing.u3_func(r, self.n_asy, self.L, self.a) * self.C
        else:
            return shrinking.u3_func(r, self.a) * self.C

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


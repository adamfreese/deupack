# shrinking.py
# Created 2026.08.18 by Adam Freese
#
# Methods to solve for numerical wave functions that approximate
# ground states of various potentials, whose large distance
# behavior becomes constant or shrinks.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

from ...constants import a0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Potential solver routine

def solve_potential(mu, pot, N=4):
    '''
    Considers a potential
        V(r) = pot.Vfun(r)
    binding two particles with a reduced mass mu, under the assumption that
        V(r) ~ r**0
    at most.  Solves for the ground state wave function using an ansatz
        u(r) = C*r*exp(-a[0]*r)*(1 + a[1]*r + a[2]*r**2 + ...)
    This method returns (a,C,E).
    ------
    Input:
    - mu ....... float
                 reduced mass (fm**-1)
    - pot ...... Potential object
                 contains Vfun as a member
                 Vfun is a float function (takes float);
                 potential energy function, as a function of r (fm**-1)
    ------
    Optional input:
    - N ........ integer
                 number of a fit parameters (default N=4)
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
    bounds = [ (-11, 11) for _ in range(N) ]
    bounds[0] = (1/(2*a0), 11) # do not allow negative a[0]
    stuff = differential_evolution(
            energy,
            bounds,
            args = (mu, pot),
            popsize = 32,
            workers = 8,
            updating = 'deferred',
            tol = 1e-5,
            maxiter = 5000
            )
    a = stuff['x']
    N2 = quad(_usq_integrand, 0, np.inf,
              args = (a, mu)
              )[0]
    C = 1/np.sqrt(N2)
    E = stuff['fun']
    return a, C, E

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wave function parametrization

def u_func(r, a):
    ''' Approximate form of the u(r) wave function.
        u(r) = C*r*exp(-a[0]*r)*(1 + a[1]*r + a[2]*r**2 + ...)
    Input:
        - r .... float or array of floats; separation (fm)
        - n .... float; highest power of r in potential
        - a .... array of floats; a[i-1] is coefficient of r**i
    Output:
        Float or array of floats with shape of r (fm**1/2)
    '''
    u = r*1 # to copy the value instead of identifying the variables
    Nmax = a.shape[0]
    for i in range(1,Nmax):
        u += a[i] * r**(i+1)
    u *= np.exp(-a[0]*r)
    return u

def u1_func(r, a):
    ''' First derivative of u. See docstring thereof for details. '''
    f0 = r*1
    f1 = 1
    Nmax = a.shape[0]
    for i in range(1,Nmax):
        f0 += a[i] * r**(i+1)
        f1 += a[i] * (i+1) * r**i
    g0 = np.exp(-a[0]*r)
    g1 = -a[0] * g0
    u1 = g0*f1 + g1*f0
    return u1

def u2_func(r, a):
    ''' Second derivative of u. See docstring thereof for details. '''
    f0 = r*1
    f1 = 1
    f2 = 0
    Nmax = a.shape[0]
    for i in range(1,Nmax):
        f0 += a[i] * r**(i+1)
        f1 += a[i] * (i+1) * r**i
        if(i > 0):
            f2 += a[i] * (i+1) * i * r**(i-1)
    g0 = np.exp(-a[0]*r)
    g1 = -a[0] * g0
    g2 = a[0]**2 * g0
    u2 = g0*f2 + 2*g1*f1 + g2*f0
    return u2

def u3_func(r, a):
    ''' Third derivative of u. See docstring thereof for details. '''
    f0 = r*1
    f1 = 1
    f2 = 0
    f3 = 0
    Nmax = a.shape[0]
    for i in range(1,Nmax):
        f0 += a[i] * r**(i+1)
        f1 += a[i] * (i+1) * r**i
        if(i > 0):
            f2 += a[i] * (i+1) * i * r**(i-1)
        if(i > 1):
            f3 += a[i] * (i+1) * i * (i-1) * r**(i-2)
    g0 = np.exp(-a[0]*r)
    g1 = -a[0] * g0
    g2 = a[0]**2 * g0
    g3 = a[0]**3 * g0
    u2 = g0*f3 + 3*g1*f2 + 3*g2*f1 + g3*f0
    return u2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper routines for the potential solver

def energy(a, mu, pot):
    ''' Expectation value of energy for a variational state.
    Input:
    - a ....... numpy array with coefficients in the variational wave function
    - mu ...... reduced mass in fm
    - pot ..... Potential object containing Vfun
    See solve_potential for further details
    '''
    num = quad(_energy_integrand, 0, np.inf,
               args = (a, mu, pot),
               )[0]
    den = quad(_usq_integrand, 0, np.inf,
               args = (a, mu),
               )[0]
    return num/den

def _energy_integrand(r, a, mu, pot):
    '''' Integrand for expected value of energy. '''
    u = u_func(r, a)
    u2 = u2_func(r, a)
    V = pot.Vfun(r)
    return u * ( V*u - u2/(2*mu))

def _usq_integrand(r, a, mu):
    ''' u**2(r) --- to find normalization. '''
    u = u_func(r, a)
    return u**2

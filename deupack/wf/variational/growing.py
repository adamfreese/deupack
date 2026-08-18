# growing.py
# Created 2026.08.18 by Adam Freese
# (split off from var_wf.py)
#
# Methods to solve for numerical wave functions that approximate
# ground states of various potentials, whose large distance
# behavior is dominated by a term r**n for n > 0.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution

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
    bounds = [ (-11, 11) for _ in range(N) ]
    stuff = differential_evolution(
            energy,
            bounds,
            args = (mu, n_asy, Vn, pot),
            popsize = 32,
            workers = 8,
            updating = 'deferred',
            tol = 1e-5,
            maxiter = 5000
            )
    a = stuff['x']
    N2 = quad(_usq_integrand, 0, np.inf,
              args = (a, mu, n_asy, Vn)
              )[0]
    C = 1/np.sqrt(N2)
    E = stuff['fun']
    return a, C, E

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wave function parametrization

def u_func(r, n, L, a):
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

def u1_func(r, n, L, a):
    ''' First derivative of u. See docstring thereof for details. '''
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

def u2_func(r, n, L, a):
    ''' Second derivative of u. See docstring thereof for details. '''
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

def u3_func(r, n, L, a):
    ''' Third derivative of u. See docstring thereof for details. '''
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
# Helper routines for the potential solver

def energy(a, mu, n_asy, Vn, pot):
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
    u = u_func(r, n_asy, L, a)
    u2 = u2_func(r, n_asy, L, a)
    V = pot.Vfun(r)
    return u * ( V*u - u2/(2*mu))

def _usq_integrand(r, a, mu, n_asy, Vn):
    ''' u**2(r) --- to find normalization. '''
    L = 2*np.sqrt(2*mu*Vn)/(n_asy+2)
    u = u_func(r, n_asy, L, a)
    return u**2

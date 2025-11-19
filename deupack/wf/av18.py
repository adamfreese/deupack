# av18.py
# Created 2025.09.29 by Adam Freese
#
# This module reads in Bob's data file for the deuteron wave function
# and creates functions for the coordinate-space S- and D-waves,
# as well as their derivatives.

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

#from numba import jit # experimental

from ..constants import kappa
from ..constants import hbar, alphaQED, mu_p, mu_n
from ..constants import mN, mp, mn, mr, Ed, mpi, mpi_0, mpi_p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables (set when make_wf is run)

# These will be filled with cubic spline objects
u_cs  = None
w_cs  = None
u1_cs = None
w1_cs = None

# Parameters for r above the interpolation range
rmax = 0 # maximum r at which the cubic spline should be used
AS = 0
AD = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wave function and derivatives

def u(r):
    ''' Radial dependence of S-wave. '''
    return u_cs(r)*(r < rmax) + AS*u_asy(r)*(r >= rmax)

def w(r):
    ''' Radial dependence of D-wave. '''
    return w_cs(r)*(r < rmax) + AD*w_asy(r)*(r >= rmax)

def u1(r):
    ''' First derivative of S-wave. '''
    return u1_cs(r)*(r < rmax) + AS*u1_asy(r)*(r >= rmax)

def u2(r):
    ''' Second derivative of S-wave.
    Basically just uses the AV18 potential and the Schrodinger equation
    to get this from u(r) and w(r).
    '''
    #return u1_cs(r,1)*(r < rmax) + AS*u2_asy(r)*(r >= rmax)
    return mN*(Ed+Vc(r))/hbar**2*u(r) + np.sqrt(8)*mN*Vt(r)*w(r)/hbar**2

def u3(r):
    ''' Third derivative of S-wave.  May become deprecated. '''
    return u1_cs(r,2)*(r < rmax) + AS*u3_asy(r)*(r >= rmax)

def w1(r):
    ''' First derivative of D-wave. '''
    return w1_cs(r)*(r < rmax) + AD*w1_asy(r)*(r >= rmax)

def w2(r):
    ''' Second derivative of D-wave.
    Basically just uses the AV18 potential and the Schrodinger equation
    to get this from u(r) and w(r).
    '''
    #return w1_cs(r,1)*(r < rmax) + AD*w2_asy(r)*(r >= rmax)
    return (mN*(Ed+Vw(r))/hbar**2 + 6/r**2)*w(r) + np.sqrt(8)*mN*Vt(r)*u(r)/hbar**2

def w3(r):
    ''' Third derivative of D-wave.  May become deprecated. '''
    return w1_cs(r,2)*(r < rmax) + AD*w3_asy(r)*(r >= rmax)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Asymptotic forms
# These are likely to go into a shared module if other wave functions are added

def u_asy(r):
    ''' Asymptotic form of the S-wave for large r. '''
    return np.exp(-kappa*r)

def u1_asy(r):
    ''' Asymptotic form of the first derivative of the S-wave for large r. '''
    return -kappa * np.exp(-kappa*r)

def u2_asy(r):
    ''' Asymptotic form of the second derivative of the S-wave for large r. '''
    return kappa**2 * np.exp(-kappa*r)

def u3_asy(r):
    ''' Asymptotic form of the third derivative of the S-wave for large r. '''
    return -kappa**3 * np.exp(-kappa*r)

def w_asy(r):
    ''' Asymptotic form of the D-wave for large r. '''
    r_ = r + 1e-9 # to division by zero
    result = np.exp(-kappa*r)*(1 + 3/(kappa*r_) + 3/(kappa*r_)**2)
    return result

def w1_asy(r):
    ''' Asymptotic form of the first derivative of the D-wave for large r. '''
    r_ = r + 1e-9 # to division by zero
    result = -kappa * np.exp(-kappa*r)*(
            1 + 3/(kappa*r_) + 6/(kappa*r_)**2 + 6/(kappa*r_)**3
            )
    return result

def w2_asy(r):
    ''' Asymptotic form of the second derivative of the D-wave for large r. '''
    r_ = r + 1e-9 # to division by zero
    result = kappa**2 * np.exp(-kappa*r)*(
            1 + 3/(kappa*r_) + 9/(kappa*r_)**2 + 18/(kappa*r_)**3 + 18/(kappa*r_)**4
            )
    return result

def w3_asy(r):
    ''' Asymptotic form of the third derivative of the D-wave for large r. '''
    r_ = r + 1e-9 # to division by zero
    result = -kappa**3 * np.exp(-kappa*r)*(
            1 + 3/(kappa*r_) + 12/(kappa*r_)**2 + 36/(kappa*r_)**3
            + 72/(kappa*r_)**4 + 72/(kappa*r_)**5
            )
    return result

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper routines

def read_wf_data():
    ''' Read wave function data from the table Bob provided.
    (Currently just r space. I'll make a k-space one when/if that's called for.)
    '''
    path = Path(__file__).parent.parent / 'data/av18r.csv'
    df = pd.read_csv(path, sep='\s+', comment='#')
    return df

def make_wf():
    ''' Sets variables and creates cubic spline objects from the data in
    Bob's wave function tables, so that the methods in this module
    can be used to estimate the deuteron wave function.
    '''
    # Global variables that will be filled by this method
    global rmax
    global AS
    global AD
    global u_cs
    global w_cs
    global u1_cs
    global w1_cs
    # First, read in the data
    df = read_wf_data()
    # Next, add a row for r=0
    df0 = pd.DataFrame({
        'r' : 0.0, 'u' : 0.0, 'w' : 0.0, 'dw/dr' : 0.0,
        'du/dr' : df['u'][0] / df['r'][0]
        }, index=[0])
    df = pd.concat([df0, df]).reset_index(drop=True)
    # Helpful to have these arrays instead of needing to call df
    r = df['r'].to_numpy()
    u = df['u'].to_numpy()
    w = df['w'].to_numpy()
    # Third, interpolate the data
    u_cs  = CubicSpline(r, u)
    w_cs  = CubicSpline(r, w)
    u1_cs = CubicSpline(r, df['du/dr'])
    w1_cs = CubicSpline(r, df['dw/dr'])
    # Make note of the maximum r for the splines
    rmax = r.max()
    # Now, get constants for asymptotic behavior at very large r
    AS = (u / u_asy(r))[-1]
    AD = (w / w_asy(r))[-1]
    # Done here
    return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Implementation of the AV18 potential itself
# This section is based on:
#   R.B. Wiringa, V.G.J. Stoks and R. Schiavilla,
#   Physical Review C 51 (1995) 38
# Since I only intend to get the deuteron wave function,
# I'm only implementing S=1, T=0

def F_C(r):
    ''' From Eq. (10) of AV18 paper. '''
    b = 4.27 # fm**-1; see comment below Eq. (9)
    x = r*b
    return 1 - (1 + 11/16*x + 3/16*x**2 + x**3/48)*np.exp(-x)

def F_delta(r):
    ''' From Eq. (10) of AV18 paper. '''
    b = 4.27 # fm**-1; see comment below Eq. (9)
    x = r*b
    return b**3*(1 + x + x**2/3)/16*np.exp(-x)

def F_t(r):
    ''' From Eq. (10) of AV18 paper. '''
    b = 4.27 # fm**-1; see comment below Eq. (9)
    x = r*b
    return 1 - (1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/144)*np.exp(-x)

def F_ls(r):
    ''' From Eq. (10) of AV18 paper. '''
    b = 4.27 # fm**-1; see comment below Eq. (9)
    x = r*b
    return 1 - (1 + x + x**2/2 + 7/48*x**3 + x**4/48)*np.exp(-x)

def F_np(r):
    ''' From Eq. (14) of AV18 paper. '''
    b = 4.27 # fm**-1; see comment below Eq. (9)
    x = r*b
    return b**2*(15*x*(1+x) + 6*x**3 + x**4)/384*np.exp(-x)

def V_C1_np(r):
    ''' From Eq. (12) of AV18 paper. '''
    beta_n = 0.0189 # fm**2; see comment below Eq. (13)
    return alphaQED*beta_n*F_np(r)/r * hbar # convert to GeV

def V_MM_np_c(r):
    ''' From Eq. (15) of AV18 paper.
    The L.A term is neglected.
    This is the central potential part of V_MM_np.
    '''
    return -alphaQED*mu_p*mu_n/(4*mn*mp) * 2/3*F_delta(r) * hbar**3

def V_MM_np_t(r):
    ''' From Eq. (15) of AV18 paper.
    The L.A term is neglected.
    This is the tensor force part of V_MM_np.
    '''
    return -alphaQED*mu_p*mu_n/(4*mn*mp) * F_t(r) / r**3 * hbar**3

def V_MM_np_ls(r):
    ''' From Eq. (15) of AV18 paper.
    The L.A term is neglected.
    This is the spin-orbit coupling part of V_MM_np.
    '''
    return -alphaQED*mu_n/(2*mn*mr) * F_ls(r) / r**3 * hbar**3

def Y_mu(r, mu):
    ''' Regularized Yukawa potential; see Eq. (19) of AV18 paper. '''
    c = 2.1 # fm**-2; see Table II
    if(r==0):
        return 0
    return np.exp(-mu*r)/(mu*r)*(1 - np.exp(-c*r**2))

def T_mu(r, mu):
    ''' Regularized tensor potential; see Eq. (19) of AV18 paper. '''
    c = 2.1 # fm**-2; see Table II
    if(r==0):
        return 0
    return np.exp(-mu*r)/(mu*r)*(1 - np.exp(-c*r**2))**2*(
            1 + 3/(mu*r) + 3/(mu*r)**2
            )

def V_pi_np_c(r):
    ''' One-pion exchange potential; see Eq. (17) of the AV18 paper.
    This is the central potential part of V_pi_np.
    Using fpp = -fnn = fc, as suggested after Eq. (19).
    '''
    fc2 = 0.075 # see text after Eq. (19)
    term1 = -fc2*(mpi_0/mpi_p)**2 * mpi_0/3 * Y_mu(r, mpi_0/hbar)
    term2 = -2*fc2 * mpi_p/3 * Y_mu(r, mpi_p/hbar)
    return term1 + term2

def V_pi_np_t(r):
    ''' One-pion exchange potential; see Eq. (17) of the AV18 paper.
    This is the tensor force part of V_pi_np.
    Using fpp = -fnn = fc, as suggested after Eq. (19).
    Also just using the average pion mass.
    '''
    fc2 = 0.075 # see text after Eq. (19)
    term1 = -fc2*(mpi_0/mpi_p)**2 * mpi_0/3 * T_mu(r, mpi_0/hbar)
    term2 = -2*fc2 * mpi_p/3 * T_mu(r, mpi_p/hbar)
    return term1 + term2

def W(r):
    ''' Woods-Saxon function; see Eq. (22) of the AV18 paper. '''
    r0 = 0.5 # fm; see Table II
    a  = 0.2 # fm; see Table II
    # For insanely large r, just return 0. Avoids an overflow warning.
    if(r > 100):
        return 0
    return 1/(1+np.exp((r-r0)/a))

def V_short_form(r, I, P, Q, R):
    ''' The short-range potential form given in Eq. (21).
    The AV18 paper's parameters (Table II) are given in MeV.
    I thus divide by 1000 to convert to GeV, which I use in this code.
    '''
    mu = mpi/hbar # average pion mass in fm**-1
    result = I*(T_mu(r,mu))**2 + (P + mu*r*Q + (mu*r)**2*R)*W(r)
    return result/1000 # convert from MeV to GeV

def V_short_c(r):
    ''' Central part of the short-range force.
    S=1, T0. Parameters from Table II of the AV18 paper.
    '''
    I = -8.62770
    P = 2605.2682
    Q = 1459.7345
    R = 441.9733
    return V_short_form(r, I, P, Q, R)

def V_short_l2(r):
    ''' L**2 part of the short-range force.
    S=1, T0. Parameters from Table II of the AV18 paper.
    '''
    I = -0.13201
    P = 253.4350
    Q = 137.4144
    R = -1.0076
    return V_short_form(r, I, P, Q, R)

def V_short_t(r):
    ''' Tensor force part of the short-range force.
    S=1, T0. Parameters from Table II of the AV18 paper.
    '''
    I = 1.485601
    P = 0
    Q = -1126.8359
    R = 370.1324
    return V_short_form(r, I, P, Q, R)

def V_short_ls(r):
    ''' Spin-orbit part of the short-range force.
    S=1, T0. Parameters from Table II of the AV18 paper.
    '''
    I = 0.10180
    P = 86.0658
    Q = 46.6655
    R = -356.5175
    return V_short_form(r, I, P, Q, R)

def V_short_ls2(r):
    ''' Spin-orbit squared part of the short-range force.
    S=1, T0. Parameters from Table II of the AV18 paper.
    '''
    I = 0.07357
    P = -217.5791
    Q = -117.9731
    R = 18.3955
    return V_short_form(r, I, P, Q, R)

def Vc(r):
    ''' Central force part of the potential in the spin-one isosinglet channel. '''
    return V_C1_np(r) + V_MM_np_c(r) + V_pi_np_c(r) + V_short_c(r)

def Vl2(r):
    ''' L**2 part of the potential in the spin-one isosinglet channel. '''
    return V_short_l2(r)

def Vt(r):
    ''' Tensor force part of the potential in the spin-one isosinglet channel. '''
    return V_MM_np_t(r) + V_pi_np_t(r) + V_short_t(r)

def Vls(r):
    ''' Spin-orbit part of the potential in the spin-one isosinglet channel. '''
    return V_MM_np_ls(r) + V_short_ls(r)

def Vls2(r):
    ''' Spin-orbit squared part of the potential in the spin-one isosinglet channel. '''
    return V_short_ls2(r)

def Vw(r):
    ''' An effective central potential for the D-wave.
    In the spin-triplet isosinglet channel, the coupled equations can be written:
        u''(r) = mN*(Ed + Vc(r))/hbar**2 * u(r) + np.sqrt(8)*mN*Vt(r) * w(r)
        w''(r) = (mN*(Ed + Vw(r))/hbar**2 + 6/r**2) * w(r) + np.sqrt(8)*mN*Vt(r) * u(r)
    This is really helpful for rewriting second derivatives in terms of u(r)
    and w(r) themselves, instead of needing to do numerical derivatives.
    '''
    return Vc(r) + 6*Vl2(r) - 3*Vls(r) + 9*Vls2(r) - 2*Vt(r)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the wave function maker on initialization

make_wf()

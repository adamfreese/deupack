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

from ..constants import kappa

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
    ''' Second derivative of S-wave. '''
    return u1_cs(r,1)*(r < rmax) + AS*u2_asy(r)*(r >= rmax)

def u3(r):
    ''' Third derivative of S-wave. '''
    return u1_cs(r,2)*(r < rmax) + AS*u3_asy(r)*(r >= rmax)

def w1(r):
    ''' First derivative of D-wave. '''
    return w1_cs(r)*(r < rmax) + AD*w1_asy(r)*(r >= rmax)

def w2(r):
    ''' Second derivative of D-wave. '''
    return w1_cs(r,1)*(r < rmax) + AD*w2_asy(r)*(r >= rmax)

def w3(r):
    ''' Third derivative of D-wave. '''
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
# Run the wave function maker on initialization

make_wf()

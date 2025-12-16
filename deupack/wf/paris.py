# paris.py
# Created 2025.11.05 by Alan Sosa
#
# This module reads in parameters for a sum-of-yukawas parametrization

import numpy as np
import pandas as pd
from pathlib import Path

from . import yukawa

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables (set when get_params is run)

# These will be filled with numpy arrays
C_J=None
D_J=None
m_j=None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use the Yukawa form for the wave function

def u(r):
    return yukawa.u(r, C_J, m_j)

def u1(r):
    return yukawa.u1(r, C_J, m_j)

def u2(r):
    return yukawa.u2(r, C_J, m_j)

def u3(r):
    return yukawa.u3(r, C_J, m_j)

def w(r):
    return yukawa.w(r, D_J, m_j)

def w1(r):
    return yukawa.w1(r, D_J, m_j)

def w2(r):
    return yukawa.w2(r, D_J, m_j)

def w3(r):
    return yukawa.w3(r, D_J, m_j)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter-reading methods

def read_params_data():
    ''' Read parameters used for analytic form of Paris wavefunction.
    (Currently just r space. I'll make a k-space one when/if that's called for.)
    '''
    path = Path(__file__).parent.parent / 'data/paris_parameters.csv'
    # Read the CSV file with pandas using sep instead of delim_whitespace
    df = pd.read_csv(path, skiprows=4, sep=r'\s+')
    # Convert to numpy array and handle the missing D_J values
    C_Js = df['C_J'].to_numpy()
    D_Js = df['D_J'].to_numpy()  # Fill NaN values with 0
    # Combine into a single array with 2 columns
    params = np.column_stack((C_Js, D_Js))
    return params

def get_params():
    global C_J, D_J, m_j  # Declare globals
    # Read data
    params = read_params_data()
    C_J= params[:,0]
    D_J= params[:,1]
    alpha = 0.23162461
    n_mj = len(C_J)
    m_j = alpha + np.arange(n_mj)  
    return 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the wave function maker on initialization

get_params()

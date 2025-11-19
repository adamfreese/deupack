# paris.py
# Created 2025.11.05 by Alan Sosa
#
# This module reads in parameters 

import numpy as np
import pandas as pd
from pathlib import Path

# Global variables (set when make_wf is run)

# These will be filled with numpy arrays
C_J=None
D_J=None
m_j=None

def u(r):
    s = C_J * np.exp(-m_j*r)
    s = np.sum(s)
    return s

def u1(r):
    s1 = -C_J*m_j * np.exp(-m_j*r)
    s1 = np.sum(s1)
    return s1

def u2(r):
    s2 = C_J*m_j**2 * np.exp(-m_j*r)
    s2 = np.sum(s2)
    return s2

def u3(r):
    s3 = -C_J*m_j**3 * np.exp(-m_j*r)
    s3 = np.sum(s3)
    return s3

def w(r):
    kr = m_j * r
    d = D_J * np.exp(-kr) * (1. + 3./kr + 3./ kr**2)
    d = np.sum(d)
    return d

def w1(r):
    kr = m_j * r
    d1 = -D_J*m_j * np.exp(-kr) * (
            1. + 3./kr + 6./ kr**2 + 6./ kr**3
        )
    d1 = np.sum(d1)
    return d1

def w2(r):
    kr = m_j * r
    d2 = D_J * m_j**2 * np.exp(-kr) * (
            1. + 3./kr + 9./ kr**2 + 
            18./ kr**3 + 18./ kr**4
        )
    d2 = np.sum(d2)
    return d2

def w3(r):
    kr = m_j * r
    d3 = -D_J * m_j**3 * np.exp(-kr) * (
            1. + 3./kr + 12./ kr**2 + 
            36./ kr**3 + 72./ kr**4 + 72./ kr**5
        )
    d3 = np.sum(d3)
    return d3


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
    #n_mj = 15
    n_mj = len(C_J)
    m_j = alpha + np.arange(n_mj)  

    return 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the wave function maker on initialization

get_params()




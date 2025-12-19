# wim.py
# Created 2025.10.10 by Adam Freese
#
# This module reads in Wim's data file for his EMT calculations,
# from a light cone convolution model, and converts them to
# the MFFs defined in our more recent non-relativistic work.
#
# The form factors this module looks at are from:
#   Adam Freese and Wim Cosyn
#   Physical Review D 106 (2022) 114013
#   Freese:2022yur

import numpy as np
import pandas as pd
from pathlib import Path

from ..constants import Md

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_wimffs():
    ''' Convert the data Wim provided into the MFFs used in our
    more recent, non-relativistic work.
    '''
    df = read_emt_data()
    t = df['t']
    A11 = df['A_++']
    A00 = df['A_00']
    Amp = df['A_-+']
    J11 = df['J_++']
    D11 = df['tD_++'] / t
    D00 = df['tD_00'] / t
    Dmp = df['tD_-+'] / t
    # New form factors
    AU = A11 + Amp/3
    AT = 4*Md**2/t*Amp
    J = J11
    DU = 2/3*D11 + (D00 + t/(4*Md**2)*(Dmp-D11))/(1+t/(4*Md**2))/3
    # Form factors with potential G7 contamination
    # The method of calculating and removing G7 is questionable,
    # so it's turned off by default
    DT1 = 4*Md**2/t**2*df['tD_-+']
    DT2 = (D00 - (1+t/(2*Md**2))*D11 - Dmp) / (1+t/(4*Md**2)) / 2
    new_df = pd.DataFrame({
        'Delta2' : -t,
        'AU'     : AU,
        'AT'     : AT,
        'J'      : J,
        'DU'     : DU,
        'DT1'    : DT1,
        'DT2'    : DT2
        })
    # Cull the DataFrame to use Delta2 > 1e-4,
    # because of numerical instability in Wim's code.
    culled_df = new_df[new_df['Delta2'] > 1e-4]
    return culled_df

def read_emt_data():
    ''' Read EMT data from the table Wim provided. '''
    path = Path(__file__).parent.parent / 'data/wimff.csv'
    df = pd.read_csv(path, comment='#')
    return df

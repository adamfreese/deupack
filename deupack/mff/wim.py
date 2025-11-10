# wimff.py
# Created 2025.10.10 by Adam Freese
#
# This module reads in Wim's data file for his EMT calculations,
# from a light cone convolution model, and converts them to
# the MFFs defined in our more recent non-relativistic work.

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
    A00 = df['A_++']
    Amp = df['A_-+']
    J11 = df['J_++']
    D11 = df['tD_++'] / t
    Dmp = df['tD_-+'] / t
    #
    AU = A11 + Amp/3
    AT = 4*Md**2/t*Amp
    J = J11
    DT1 = 4*Md**2/t*Dmp
    DT2 = (2*A11 - 4*J11) + 4*Md**2/t*(A11 - A00)
    DU = D11 + Dmp/3 + DT2/3
    new_df = pd.DataFrame({
        'Delta2' : -t,
        'AU'     : AU,
        'AT'     : AT,
        'J'      : J,
        'DU'     : DU,
        'DT1'    : DT1,
        'DT2'    : DT2
        })
    return new_df

def read_emt_data():
    ''' Read EMT data from the table Wim provided. '''
    path = Path(__file__).parent.parent / 'data/wimff.csv'
    df = pd.read_csv(path, comment='#')
    return df

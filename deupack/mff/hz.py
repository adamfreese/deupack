# hezahed.py
# Created 2025.10.10 by Adam Freese
#
# This module reads in Fangcheng's data file for his GFF calculations,
# and converts them to the MFFs in Cosyn/Freese/Sosa.
#
# The form factors this module looks at are from:
#   Fancheng He and Ismail Zahed
#   Physical REview C 110 (2024) 014312
#   He:2024vzz

import numpy as np
import pandas as pd
from pathlib import Path

from ..constants import Md

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_hzmffs():
    ''' Convert He and Zahed's MFFs into those of Cosyn, Freese and Sosa. '''
    df = read_mffs()
    t = df['t']
    A = df['A']
    Q = df['Q']
    J = df['J']
    D0 = df['D0']
    D2 = df['D2']
    D3 = df['D3']
    AU = A
    AT = -Q
    DU = D0
    DT2 = 2*D2
    DT1 = 2*Md**2/t * D3
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

def read_mffs():
    ''' Read EMT data from the tables Fangcheng provided. '''
    path = Path(__file__).parent.parent / 'data/hz'
    df_A =  pd.read_csv(path / "data_A.txt",  header=None, sep='\s+')
    df_J =  pd.read_csv(path / "data_J.txt",  header=None, sep='\s+')
    df_Q =  pd.read_csv(path / "data_Q.txt",  header=None, sep='\s+')
    df_D0 = pd.read_csv(path / "data_D0.txt", header=None, sep='\s+')
    df_D2 = pd.read_csv(path / "data_D2.txt", header=None, sep='\s+')
    df_D3 = pd.read_csv(path / "data_D3.txt", header=None, sep='\s+')
    t = -df_A[0]
    A = df_A[1]
    J = df_J[1]
    Q = df_Q[1]
    D0 = df_D0[1]
    D2 = df_D2[1]
    D3 = df_D3[1]
    df = pd.DataFrame({
        't' : t,
        'A' : A,
        'Q' : Q,
        'J' : J,
        'D0' : D0,
        'D2' : D2,
        'D3' : D3
        })
    return df

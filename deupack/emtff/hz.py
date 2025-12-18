# hezahed.py
# Created 2025.10.10 by Adam Freese
#
# This module reads in Fangcheng's data file for his GFF calculations,
# and converts them to the EMTFFs in Cosyn/Freese/Sosa.
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

def make_hzffs(impulse_approximation=False):
    ''' Convert He and Zahed's GFFs into those of Cosyn, Freese and Sosa. '''
    df = read_emtffs(impulse_approximation=impulse_approximation)
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
    DT2 = D2
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

def read_emtffs(impulse_approximation=False):
    ''' Read EMT data from the tables Fangcheng provided.
    ----------
    Input:
        impulse_approximation : bool, optional
            if True, use the impulse approximation (IA) tables
    Output:
        pandas.DataFrame
    '''
    path = Path(__file__).parent.parent / 'data/hz'
    def gff_path(name):
        if(impulse_approximation):
            return path / "data_{}_IA.txt".format(name)
        else:
            return path / "data_{}.txt".format(name)
    df_A =  pd.read_csv(gff_path('A'),  header=None, sep='\s+')
    df_J =  pd.read_csv(gff_path('J'),  header=None, sep='\s+')
    df_Q =  pd.read_csv(gff_path('Q'),  header=None, sep='\s+')
    df_D0 = pd.read_csv(gff_path('D0'), header=None, sep='\s+')
    df_D2 = pd.read_csv(gff_path('D2'), header=None, sep='\s+')
    df_D3 = pd.read_csv(gff_path('D3'), header=None, sep='\s+')
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

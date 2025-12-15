# hezahed.py
# Created 2025.10.23 by Adam Freese
#
# This module reads in Julia's data files for her GFF calculations,
# and converts them to the MFFs in Cosyn/Freese/Sosa.
#
# The form factors this module looks at are from:
#   J.Yu. Panteleeva, E. Epelbaum, A.M. Gasparyan, J. Gegelia
#   Acta Phys Polon B56 (2025) 3-A19
#   Panteleeva:2024abz

import numpy as np
import pandas as pd
from pathlib import Path

from ..constants import Md

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_peggffs():
    ''' Convert Panteleeva et al.'s MFFs into those of Cosyn, Freese and Sosa. '''
    df = read_emtffs()
    t = df['t']
    E0 = df['E0']
    E2 = df['E2']
    D0 = df['D0']
    D2 = df['D2']
    D3 = df['D3']
    J = df['J']
    AU = E0
    AT = -2*E2
    DU = -4*D0
    DT1 = 8*D3
    DT2 = -2*D2
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



def make_peggffsLO():
    ''' Convert Panteleeva et al.'s MFFs into those of Cosyn, Freese and Sosa. '''
    df = read_emtffsLO()
    t = df['t']
    E0 = df['E0']
    E2 = df['E2']
    D0 = df['D0']
    D2 = df['D2']
    D3 = df['D3']
    J = df['J']
    AU = E0
    AT = -2*E2
    DU = -4*D0
    DT1 = 8*D3
    DT2 = -2*D2
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


def read_emtffs():
    ''' Read EMT data from the tables Fangcheng provided. '''
    path = Path(__file__).parent.parent / 'data/pegg'
    # c8 and c9 provided by Julia Panteleeva (private communication)
    c8 = -2.77/1000 # MeV**-1
    c9 = 0
    # Use the highest-order data available for each MFF
    df_E0 = pd.read_csv(path / "E0_nnlo.txt", comment='#')
    df_E2 = pd.read_csv(path / "E2_nnlo.txt", comment='#')
    df_D0 = pd.read_csv(path / "D0_nlo.txt",  comment='#')
    df_D2 = pd.read_csv(path / "D2_nlo.txt",  comment='#')
    df_D3 = pd.read_csv(path / "D3_nlo.txt",  comment='#')
    df_J  = pd.read_csv(path / "J_lo.txt",    comment='#')
    t  = -(df_E0['q (MeV)']/1000)**2
    E0 = df_E0['E0'] + df_E0['E0c8']*c8 + df_E0['E0c9']*c9
    E2 = df_E2['E2'] + df_E2['E2c8']*c8 + df_E2['E2c9']*c9
    D0 = df_D0['D0'] + df_D0['D0c8']*c8
    D2 = df_D2['D2'] + df_D2['D2c8']*c8
    D3 = df_D3['D3'] + df_D3['D3c8']*c8
    J  =  df_J['J']  +  df_J['Jc9'] *c9
    df = pd.DataFrame({
        't' : t,
        'E0' : E0,
        'E2' : E2,
        'D0' : D0,
        'D2' : D2,
        'D3' : D3,
        'J' : J
        })
    return df



def read_emtffsLO():
    ''' Read EMT data from the tables Fangcheng provided. '''
    path = Path(__file__).parent.parent / 'data/pegg'
    # c8 and c9 provided by Julia Panteleeva (private communication)
    c8 = -2.77/1000 # MeV**-1
    c9 = 0
    # Use the highest-order data available for each MFF
    df_E0 = pd.read_csv(path / "E0_lo.txt", comment='#')
    df_E2 = pd.read_csv(path / "E2_lo.txt", comment='#')
    df_D0 = pd.read_csv(path / "D0_lo.txt",  comment='#')
    df_D2 = pd.read_csv(path / "D2_lo.txt",  comment='#')
    df_D3 = pd.read_csv(path / "D3_lo.txt",  comment='#')
    df_J  = pd.read_csv(path / "J_lo.txt",    comment='#')
    t  = -(df_E0['q (MeV)']/1000)**2
    E0 = df_E0['E0'] 
    E2 = df_E2['E2'] 
    D0 = df_D0['D0'] + df_D0['D0c8']*c8
    D2 = df_D2['D2'] + df_D2['D2c8']*c8
    D3 = df_D3['D3'] + df_D3['D3c8']*c8
    J  =  df_J['J']  +  df_J['Jc9'] *c9
    df = pd.DataFrame({
        't' : t,
        'E0' : E0,
        'E2' : E2,
        'D0' : D0,
        'D2' : D2,
        'D3' : D3,
        'J' : J
        })
    return df

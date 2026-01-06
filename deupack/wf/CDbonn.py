# CDbonn.py
# Created 2025.11.20 by Alan Sosa
#
# This module reads in parameters for a sum-of-yukawas parametrization

import numpy as np
import pandas as pd
from pathlib import Path

from .yukawa import dwf_yukawa

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dwf_cdbonn(dwf_yukawa):
    ''' Uses the sum-of-Yukawas parametrization of the CD Bonn wave function. '''

    def __init__(self):
        CJ, DJ, MJ = self._read_params_data()
        super().__init__(CJ, DJ, MJ)

    def _read_params_data(self):
        ''' Read parameters used for analytic form of CD Bonn wavefunction. '''
        path = Path(__file__).parent.parent / 'data/CDbonn_parameters.csv'
        # Read the CSV file with pandas using sep instead of delim_whitespace
        df = pd.read_csv(path, skiprows=4, sep=r'\s+')
        # Convert to numpy array and handle the missing D_J values
        CJ = df['C_J'].to_numpy()
        DJ = df['D_J'].to_numpy()
        alpha = 0.2315380
        n_MJ = len(CJ)
        MJ = alpha + 0.9*np.arange(n_MJ)
        return CJ, DJ, MJ

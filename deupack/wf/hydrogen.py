# hydrogen.py
# Created 2026.08.03 by Adam Freese
#
# A class for hydrogen atom wave functions.
# In progress. Currently only the ground state
# TODO

import numpy as np
from .dwf import DWF
from ..constants import hbar, alphaQED

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dwf_hydrogen(DWF):
    ''' Creates a wave function for a non-relativistic hydrogen-like atom
    and hijacks the machinery of deupack to calculate its EMT-FFs.
    '''

    def __init__(self,
                 n = 0,
                 l = 0,
                 ml = 0,
                 mN = 0.106,   # muon mass (GeV)
                 alpha = alphaQED
                 ):
        super().__init__()
        self.name = 'hydrogen_{:d}_{:d}_{:d}'.format(n,l,ml)
        self.mN = mN
        self.mNfm = mN / hbar
        self.alpha = alpha
        # Quantum numbers
        self.n = n
        self.l = l
        self.ml = ml
        # Currently just ground state
        # TODO: general states
        if(n!=0 or l!=0 or ml!=0):
            raise NotImplementedError("Currently only n=l=ml=0 implemented.")
        return

    # Wave function overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        factor = np.sqrt(self.alpha**3*self.mNfm**3/2)
        kappa = self.alpha*self.mNfm/2
        return factor * r * np.exp(-kappa*r)

    def u1(self, r):
        factor = np.sqrt(self.alpha**3*self.mNfm**3/2)
        kappa = self.alpha*self.mNfm/2
        return factor * (1 - kappa*r) * np.exp(-kappa*r)

    def u2(self, r):
        factor = np.sqrt(self.alpha**3*self.mNfm**3/2)
        kappa = self.alpha*self.mNfm/2
        return factor * (-2*kappa + kappa**2*r) * np.exp(-kappa*r)

    def u3(self, r):
        factor = np.sqrt(self.alpha**3*self.mNfm**3/2)
        kappa = self.alpha*self.mNfm/2
        return factor * (3*kappa**2 - kappa**3*r) * np.exp(-kappa*r)

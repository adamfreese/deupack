# mab.py
# Created 2025.11.13 by Alan Sosa

from ...constants import mN, hbar

from .nff import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_mab(nff_with_SN):
    ''' Nucleon EMT-FFs from the meson dominance model of:
        Masjuan, Ruiz Arriola and Broniowski
        Phys. Rev. D 87 (2013) 014005
        Masjuan:2012sk
    '''

    def __init__(self):
        super().__init__()
        self.name = "mab"
        return

    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN(self, k):
        ''' AN. Assumes k is in GeV. '''
        mf2_1270 = 1.27    # f2 1270 mass (GeV)
        mf2_1430 = 1.43    # f2 1430 mass (GeV)
        a = self.dipole(k, mf2_1270)*self.dipole(k, mf2_1430)
        return a

    def JN(self, k):
        ''' JN. Assumes k is in GeV. '''
        mf2_1270 = 1.27    # f2 1270 mass (GeV)
        mf2_1430 = 1.43    # f2 1430 mass (GeV)
        a = 0.5*self.dipole(k, mf2_1270)*self.dipole(k, mf2_1430)
        return a

    def DN(self, k):
        ''' DN. Assumes k is in GeV. '''
        mf2_1270 = 1.27    # f2 1270 mass (GeV)
        mf2_1430 = 1.43    # f2 1430 mass (GeV)
        msigma   = 0.8     # sigma mass (GeV)
        a = -2.*self.dipole(k, mf2_1270)*self.dipole(k, mf2_1430)*self.dipole(k, msigma)
        return a

    def mass_radius_squared(self):
        mf2_1270 = 1.27    # f2 1270 mass (GeV)
        mf2_1430 = 1.43    # f2 1430 mass (GeV)
        dAdt = 2/mf2_1270**2 + 2/mf2_1430**2
        return 6 * dAdt * hbar**2

    # Auxiliary functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dipole(self, k, Lambda):
        ''' Dipole form. '''
        return 1. / (1 + (k/Lambda)**2)**2

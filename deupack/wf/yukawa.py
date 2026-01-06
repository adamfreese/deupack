# yukawa.py
# Created 2025.12.15 by Adam Freese to consolidate codes by Alan Sosa
#
# This parametrizes the deuteron wave function as a sum of Yukawa-type terms,
# in line with the suggestion in:
#   Lacombe, Loiseau, Vinh Mau, Cote, Pires, and de Tourreil,
#   Parametrization of the deuteron wave function of the Paris N-N potential
#   Phys.Lett.B 101 (1981) 139-140
#   Lacombe:1981eg

import numpy as np

from .dwf import DWF

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dwf_yukawa(DWF):
    ''' Parametrizes the deuteron wave function as a sum of Yukawa-type terms.
    It should be initialized using three arrays of CJ, DJ and MJ values,
    following the parametrization in:
       Lacombe, Loiseau, Vinh Mau, Cote, Pires, and de Tourreil,
       Parametrization of the deuteron wave function of the Paris N-N potential
       Phys.Lett.B 101 (1981) 139-140
       Lacombe:1981eg
    '''

    def __init__(self, CJ, DJ, MJ):
        super().__init__()
        self.CJ = CJ
        self.DJ = DJ
        self.MJ = MJ
        return

    # S-wave ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def u(self, r):
        s = self.CJ * np.exp(-self.MJ*r)
        s = np.sum(s)
        return s

    def u1(self, r):
        s1 = -self.CJ*self.MJ * np.exp(-self.MJ*r)
        s1 = np.sum(s1)
        return s1

    def u2(self, r):
        s2 = self.CJ*self.MJ**2 * np.exp(-self.MJ*r)
        s2 = np.sum(s2)
        return s2

    def u3(self, r):
        s3 = -self.CJ*self.MJ**3 * np.exp(-self.MJ*r)
        s3 = np.sum(s3)
        return s3

    # D-wave ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def w(self, r):
        kr = self.MJ * r
        d = self.DJ * np.exp(-kr) * (1. + 3./kr + 3./ kr**2)
        d = np.sum(d)
        return d

    def w1(self, r):
        kr = self.MJ * r
        d1 = -self.DJ*self.MJ * np.exp(-kr) * (
                1. + 3./kr + 6./ kr**2 + 6./ kr**3
            )
        d1 = np.sum(d1)
        return d1

    def w2(self, r):
        kr = self.MJ * r
        d2 = self.DJ * self.MJ**2 * np.exp(-kr) * (
                1. + 3./kr + 9./ kr**2 +
                18./ kr**3 + 18./ kr**4
            )
        d2 = np.sum(d2)
        return d2

    def w3(self, r):
        kr = self.MJ * r
        d3 = -self.DJ * self.MJ**3 * np.exp(-kr) * (
                1. + 3./kr + 12./ kr**2 +
                36./ kr**3 + 72./ kr**4 + 72./ kr**5
            )
        d3 = np.sum(d3)
        return d3

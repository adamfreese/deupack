# mit.py
# Created 2025.09.30 by Adam Freese

from ...constants import mN, hbar

from .nff import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_mit(nff_with_SN):
    '''
    This class contains formulas for the EMT form factors of a nucelon,
    as calculated by:
      Daniel C. Hackett, Dimitra A. Pefkou, and Phiala E. Shanahan
      Physical Review Letters 132 (2024) 241904
      Hackett:2023rif
    which I will refer to as MIT after the authors' institution.
    
    MIT actually give two parametric fits to their lattice data:
    (1) a dipole fit, and (2) a z-expansion.
    Both fits achieve qualitatively similar results and similar chi2/dof.
    For simplicity, I use their dipole forms here.
    
    One minor tweak I make here is that I enforce the sum rules A(0)=1 and J(0)=1/2.
    I adjust the alpha parameters for Ag and Jg to achieve this.
    On the other hand, I use the central fit values for Aq and Jq.
    Since D(0) is not constrained by any conservation laws,
    I use the central values reported by MIT for both Dq(0) and Dg(0)
    
    Parameters of MIT are given in Table III of their supplemental material.
    '''

    def __init__(self):
        super().__init__()
        return

    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN(self, k):
        ''' Form factor AN. Assumes k is in GeV. '''
        return self.AN_q(k) + self.AN_g(k)

    def JN(self, k):
        ''' Form factor JN. Assumes k is in GeV. '''
        return self.JN_q(k) + self.JN_g(k)

    def DN(self, k):
        ''' Form factor DN. Assumes k is in GeV. '''
        return self.DN_q(k) + self.DN_g(k)

    def cN(self, k):
        ''' Form factor cN. Assumes k is in GeV.
        This was not found by the MIT group, but we use a phenomenological
        estimate from
            Lorcé, Moutarde and Trawiński
            European Physical Journal C 79 (2019) 89
            Lorce:2018egm
        as a placeholder.

        The quark and gluon pieces are designed to cancel, so this method
        will by default give zero. However, the quark and gluon methods are
        defined independently, so can be independently overridden. It is thus
        possible for this method to give non-zero results in derived classes
        that for instance set all quark contributions or all gluon contributions
        to zero.
        '''
        return self.cN_q(k) + self.cN_g(k)

    def mass_radius_squared(self):
        ''' Derivatives of dipole forms, summed in quadrature. '''
        alpha_q  =  0.510 # reported 0.510(25)
        Lambda_q =  1.477 # reported 1.477(44)
        alpha_g  =  0.490 # reported 0.501(27); adjusted for sum rule
        Lambda_g =  1.262 # reported 1.262(18)
        dAdt_q = 2*alpha_q/Lambda_q**2
        dAdt_g = 2*alpha_g/Lambda_g**2
        return 6 * (dAdt_q + dAdt_g) * hbar**2

    # Auxiliary functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        ''' Quark contribution to AN. Assumes k is in GeV. '''
        alpha  =  0.510 # reported 0.510(25)
        Lambda =  1.477 # reported 1.477(44)
        return self.dipole(k, alpha, Lambda)

    def JN_q(self, k):
        ''' Quark contribution to JN. Assumes k is in GeV. '''
        alpha  =  0.251 # reported 0.251(21)
        Lambda =  1.62  # reported 1.62(13)
        return self.dipole(k, alpha, Lambda)

    def DN_q(self, k):
        ''' Quark contribution to DN. Assumes k is in GeV. '''
        alpha  = -1.30 # reported -1.30(49)
        Lambda =  0.81 # reported 0.81(14)
        return self.dipole(k, alpha, Lambda)

    def AN_g(self, k):
        ''' Gluon contribution to AN. Assumes k is in GeV. '''
        alpha  =  0.490 # reported 0.501(27); adjusted for sum rule
        Lambda =  1.262 # reported 1.262(18)
        return self.dipole(k, alpha, Lambda)

    def JN_g(self, k):
        ''' Gluon contribution to JN. Assumes k is in GeV. '''
        alpha  =  0.249 # reported 0.255(13); adjusted for sum rule
        Lambda =  1.399 # reported 1.399(49)
        return self.dipole(k, alpha, Lambda)

    def DN_g(self, k):
        ''' Gluon contribution to DN. Assumes k is in GeV. '''
        alpha  = -2.57  # reported -2.57(84)
        Lambda =  0.538 # reported 0.538(65)
        return self.dipole(k, alpha, Lambda)

    def cN_q(self, k):
        ''' Quark contribution to cN. Assumes k is in GeV.
        Dipole form, using parameters from:
            Lorcé, Moutarde and Trawiński
            European Physical Journal C 79 (2019) 89
            Lorce:2018egm
        '''
        c0 = -0.11 # EPJC 79 (2019) 89
        L  =  0.91 # EPJC 79 (2019) 89
        return self.dipole(k, c0, L)

    def cN_g(self, k):
        ''' Gluon contribution to cN. Assumes k is in GeV.
        Dipole form, using parameters from:
            Lorcé, Moutarde and Trawiński
            European Physical Journal C 79 (2019) 89
            Lorce:2018egm
        c0 is flipped by a sign so this cancels the quark contribution.
        '''
        c0 =  0.11 # EPJC 79 (2019) 89
        L  =  0.91 # EPJC 79 (2019) 89
        return self.dipole(k, c0, L)

    def dipole(self, k, alpha, Lambda):
        ''' Dipole form. '''
        return alpha / (1 + (k/Lambda)**2)**2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_mit_quark(nff_mit):
    ''' A quark-only variation on nff_mit. '''

    def __init__(self):
        super().__init__()
        return

    # Overrides to eliminate gluons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_g(self, k):
        return k*0

    def JN_g(self, k):
        return k*0

    def DN_g(self, k):
        return k*0

    def cN_g(self, k):
        return k*0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_mit_gluon(nff_mit):
    ''' A gluon-variation on nff_mit. '''

    def __init__(self):
        super().__init__()
        return

    # Overrides to eliminate quarks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        return k*0

    def JN_q(self, k):
        return k*0

    def DN_q(self, k):
        return k*0

    def cN_q(self, k):
        return k*0

    def SN(self, k):
        # Note that gluons cannot contribute to SN
        return k*0

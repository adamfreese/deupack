# nff.py
# Created 2026.02.03 by Adam Freese
#
# A base class specifying the interface for nucleon EMT-FFs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NFF:
    ''' An empty class. This just defines the expected interface of any set of
    nucleon EMT form factors. This ensures that an object in any derived class
    will have the expected functions, even if the derived class doesn't define
    them (in which case they will just return 0).
    '''

    def __init__(self):
        self.name = "" # every nff class should have a name
        return

    # The expected form factors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN(self, k):
        return k*0

    def JN(self, k):
        return k*0

    def DN(self, k):
        return k*0

    def SN(self, k):
        return k*0

    def cN(self, k):
        return k*0

    # Other expected routines ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mass_radius_squared(self):
        return 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_with_SN(NFF):
    ''' An otherwise empty class that fills SN with a reasonable dipole form.
    This class exists because nearly all the work on nucleon EMT-FFs only
    deals with the symmetric part of the EMT, and something else is needed
    for the antisymmetric form factor.
    '''

    def __init__(self):
        super().__init__()
        return

    def SN(self, k):
        ''' Dipole form, assuming axial vector meson dominance,
        and using the JAM value for total quark spin.
        The s0 value is from:
            C. Cocuzza et al.
            Physical REview D 106 (2022) L031502
            Cocuzza:2022jye
        '''
        ma1 = 1.23  # PDG
        s0  = 0.204 # JAM22; Cocuzza et al., PRD 106 (2022) L031502
        return s0 / (1 + (k/ma1)**2)**2
